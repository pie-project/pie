use crate::jit::Ctx;
use crate::jit::Launch;
use core::ffi::c_void;
use kernels::Bind;
use kernels::Fire;
use kernels::Refusal;
use kernels_macros::routine;

use crate::jit::abi::Tensor;
use crate::views::{AttnMask, Dsv4CompKvPages, KvCache, MtpPendingHidden, RecurrentState};
use kernels::raises::Struct;
use kernels::routine::{Const, In, Out};

use crate::jit::Abi;
#[allow(unused_imports)]
use crate::jit::abi::bf16;
use crate::jit::abi::f16;
use crate::rope::Yarn;
use kernels::routine::InOut;
// SEVEN NAMES STOOD HERE, and none of them was a re-export. `kv_paged`'s
// five write-KV routines and `qkv_fused`'s two were imported into this
// module so that the launcher bodies below could call them; those bodies
// were the ones the marks turned into `#[routine]`s registered from their
// own modules, and a registry entry is reached by symbol rather than by
// path. Nothing here names them any more. The routines are untouched and
// still public where they are defined.

pub mod fa2;

pub mod fa4;

pub mod plan;

pub mod xqa;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_scheme(pub u8);

impl kv_scheme {
    #[must_use]
    pub const fn of(scheme: crate::attn::KvScheme) -> Self {
        Self(scheme as i32 as u8)
    }

    #[must_use]
    pub const fn scheme(self) -> Option<crate::attn::KvScheme> {
        use crate::attn::KvScheme;

        match self.0 {
            0 => Some(KvScheme::Native),
            1 => Some(KvScheme::Fp8PerTensor),
            2 => Some(KvScheme::Int8PerTokenHead),
            3 => Some(KvScheme::Fp8PerTokenHead),
            4 => Some(KvScheme::Fp4Block),
            _ => None,
        }
    }
}

impl crate::jit::Abi for kv_scheme {
    const CPP: &'static str = "::pie::attn::KvScheme";
    const TY: kernels::Ty = kernels::Ty::KvScheme;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U8(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::U8(v) => Ok(Self(*v)),
            _ => Err(kernels::Refusal::Kind {
                at,
                want: kernels::Ty::KvScheme,
            }),
        }
    }
}

crate::arg_via_abi!(kv_scheme);

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_dtype(pub u8);

impl kv_dtype {
    #[must_use]
    pub const fn of(dtype: crate::attn::KvDType) -> Self {
        Self(dtype as i32 as u8)
    }
}

impl crate::jit::Abi for kv_dtype {
    const CPP: &'static str = "::pie::attn::KvDType";
    const TY: kernels::Ty = kernels::Ty::KvDType;
    fn arg(&self) -> crate::jit::ArgValue {
        crate::jit::ArgValue::U8(self.0)
    }
    fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
        match value {
            crate::jit::ArgValue::U8(v) => Ok(Self(*v)),
            _ => Err(kernels::Refusal::Kind {
                at,
                want: kernels::Ty::KvDType,
            }),
        }
    }
}

crate::arg_via_abi!(kv_dtype);

#[must_use]
const fn scheme_byte(n: i32) -> u8 {
    if n < 0 || n > u8::MAX as i32 {
        u8::MAX
    } else {
        n as u8
    }
}

pub mod params {
    use core::ffi::c_void;

    use kernels::Ty;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct StructuredMaskParams {
        pub kind: u32,
        pub window: u32,
        pub sink: u32,
    }

    const STRUCTURED_MASK_PARAMS: &str = "::pie::attn::StructuredMaskParams";

    impl crate::jit::Abi for *const StructuredMaskParams {
        const CPP: &'static str = "const ::pie::attn::StructuredMaskParams*";
        const TY: Ty = Ty::StructuredMasks;
        fn arg(&self) -> crate::jit::ArgValue {
            crate::jit::ArgValue::Ptr(*self as *mut c_void)
        }
        fn unpack(value: &crate::jit::ArgValue, at: usize) -> Result<Self, kernels::Refusal> {
            match value {
                crate::jit::ArgValue::Ptr(p) => Ok(p.cast::<StructuredMaskParams>().cast_const()),
                _ => Err(kernels::Refusal::Kind {
                    at,
                    want: Ty::StructuredMasks,
                }),
            }
        }
    }

    crate::arg_via_abi!(*const StructuredMaskParams);

    const _: () = assert!(
        ::core::mem::size_of::<StructuredMaskParams>() == 12,
        "StructuredMaskParams: sizeof disagrees with the measured \
         pie::attn::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::align_of::<StructuredMaskParams>() == 4,
        "StructuredMaskParams: alignof disagrees with the measured \
         pie::attn::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, kind) == 0,
        "StructuredMaskParams.kind: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::kind",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, window) == 4,
        "StructuredMaskParams.window: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::window",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, sink) == 8,
        "StructuredMaskParams.sink: offset disagrees with the measured \
         pie::attn::StructuredMaskParams::sink",
    );

    pub static LAYOUTS: &[crate::jit::Layout] = &[crate::jit::Layout {
        cpp: STRUCTURED_MASK_PARAMS,
        size: 12,
        align: 4,
        fields: &[("kind", 0), ("window", 4), ("sink", 8)],
        probe: "nvrtc-probes/attn_structured_mask.py",
    }];
}

pub mod attention_flashinfer {
    use crate::jit::{Ctx, Launch};
    use crate::routine::Fire;

    use crate::jit::abi::Tensor;
    use kernels::Refusal;
    use kernels::routine::{In, Out};
    use kernels_macros::routine;

    use kernels::Bind;

    #[routine(whole, untraced)]
    pub fn attn_score_fold_heads(
        ctx: &Ctx<'_>,
        scores: In<Tensor<f32>>,
        score_indptr: In<Tensor<i32>>,
        kv_page_indptr: In<Tensor<u32>>,
        kv_last_page_lens: In<Tensor<u32>>,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        folded: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let (scores, score_indptr, kv_page_indptr, kv_last_page_lens, folded) = (
            scores.ptr,
            score_indptr.ptr,
            kv_page_indptr.ptr,
            kv_last_page_lens.ptr,
            folded.ptr,
        );

        const FOLD_BLOCK: u32 = 256;

        const FOLD_GRID_Y: u32 = 64;

        ctx.fire(
            Fire::at(
                "attn/attention_flashinfer.cuh",
                "::pie::attn::attn_score_fold_heads",
            )
            .apply(Launch::grid(
                [num_requests.unsigned_abs(), FOLD_GRID_Y, 1],
                [FOLD_BLOCK, 1, 1],
            )),
            &[
                scores.arg(),
                score_indptr.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_size.arg(),
                num_q_heads.arg(),
                folded.arg(),
            ],
        )
    }
}

pub mod attention_score_post {
    use super::{Ctx, Launch, Refusal};

    use kernels::{Bind, Fire};

    const NORMALIZE_BLOCK: u32 = 256;

    pub const PREFILL_FOLD_GRID_Y: u32 = 32;

    #[allow(clippy::too_many_arguments)]
    pub fn attn_score_normalize(
        ctx: &Ctx<'_>,
        scores: *mut f32,
        score_indptr: *const i32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
    ) -> Result<(), Refusal> {
        ctx.fire(
            Fire::at(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_score_normalize",
            )
            .apply(Launch::grid(
                [num_requests.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                [NORMALIZE_BLOCK, 1, 1],
            )),
            &[
                scores.arg(),
                score_indptr.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_size.arg(),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_score_normalize(
        ctx: &Ctx<'_>,
        scores: *mut f32,
        score_indptr: *const i32,
        qo_indptr: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) -> Result<(), Refusal> {
        ctx.fire(
            Fire::at(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_prefill_score_normalize",
            )
            .apply(Launch::grid(
                [
                    num_requests.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                    window.unsigned_abs(),
                ],
                [NORMALIZE_BLOCK, 1, 1],
            )),
            &[
                scores.arg(),
                score_indptr.arg(),
                qo_indptr.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_size.arg(),
                window.arg(),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_score_fold(
        ctx: &Ctx<'_>,
        scores: *const f32,
        folded: *mut f32,
        score_indptr: *const i32,
        qo_indptr: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        page_size: i32,
        num_requests: i32,
        num_q_heads: i32,
        window: i32,
    ) -> Result<(), Refusal> {
        ctx.fire(
            Fire::at(
                "attn/attention_score_post.cuh",
                "::pie::attn::attn_prefill_score_fold",
            )
            .apply(Launch::grid(
                [num_requests.unsigned_abs(), PREFILL_FOLD_GRID_Y, 1],
                [NORMALIZE_BLOCK, 1, 1],
            )),
            &[
                scores.arg(),
                folded.arg(),
                score_indptr.arg(),
                qo_indptr.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                page_size.arg(),
                num_q_heads.arg(),
                window.arg(),
            ],
        )
    }
}

pub mod dsa_indexer {
    pub const K_BLOCK: u32 = 256;

    #[must_use]
    pub fn q_rope_block(n_heads: i32) -> u32 {
        let rounded = ((n_heads.max(0) + 31) / 32) * 32;
        #[allow(clippy::cast_sign_loss)]
        let block = rounded as u32;
        if block < 32 { 32 } else { block }
    }
}

pub mod page_compact {
    pub const K_BLOCK: u32 = 256;
}

#[routine(whole, untraced)]
pub fn compact_page_csr(
    ctx: &Ctx<'_>,
    keep: In<Tensor<u8>>,
    keep_stride: Const<u32>,
    kvc: In<Struct<KvCache>>,
    num_requests: Const<i32>,
    scratch_counts: Out<Tensor<u32>>,
    page_indices_out: Out<Tensor<u32>>,
    page_indptr_out: Out<Tensor<u32>>,
    last_page_lens_out: Out<Tensor<u32>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let keep_stride = *keep_stride;
    let page_indices_in = kvc.page_indices as *const u32;
    let page_indptr_in = kvc.page_indptr as *const u32;

    let last_page_lens_in = kvc.last_page_lens as *const u32;
    let scratch_counts = scratch_counts.ptr;
    let page_indices_out = page_indices_out.ptr;
    let page_indptr_out = page_indptr_out.ptr;
    let last_page_lens_out = last_page_lens_out.ptr;
    let num_requests = *num_requests;
    if scratch_counts.is_null() {
        return Err(Refusal::Absent {
            what: "the compaction scratch buffer",
        });
    }
    let launch = Launch::per_row(num_requests.unsigned_abs(), page_compact::K_BLOCK);

    ctx.fire(
        Fire::at(
            "attn/page_compact.cuh",
            "::pie::attn::count_kept<::pie::i32(256)>",
        )
        .apply(launch),
        &[
            page_indptr_in.arg(),
            keep.arg(),
            keep_stride.arg(),
            num_requests.arg(),
            scratch_counts.arg(),
        ],
    )?;
    ctx.fire(
        Fire::at(
            "attn/page_compact.cuh",
            "::pie::attn::scan_and_scatter<::pie::i32(256)>",
        )
        .apply(launch),
        &[
            page_indices_in.arg(),
            page_indptr_in.arg(),
            last_page_lens_in.arg(),
            keep.arg(),
            scratch_counts.cast_const().arg(),
            keep_stride.arg(),
            num_requests.arg(),
            page_indptr_out.arg(),
            last_page_lens_out.arg(),
            page_indices_out.arg(),
        ],
    )
}

pub mod attention_naive {
    pub const BLOCK: u32 = 256;
}

#[routine(bf16, whole, out(out = like(target_hidden)))]
pub fn mtp_shift_hidden<T>(
    ctx: &Ctx<'_>,
    target_hidden: In<Tensor<T>>,
    pending_hidden: In<Tensor<T>>,
    out: Out<Tensor<T>>,
    qo_indptr: In<Tensor<i32>>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal>
where
    <T as kernels::Elem>::Read: Abi + kernels::Bind<crate::jit::ArgValue>,
    <T as kernels::Elem>::Write:
        Abi + kernels::Bind<crate::jit::ArgValue> + kernels::BindMut<crate::jit::ArgValue>,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    // The request count is the CSR operand's own row count.
    let num_requests = qo_indptr.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_ids = rsv.slots;

    if matches!(pending_hidden.ptr.arg(), crate::jit::ArgValue::Ptr(p) if p.is_null()) {
        return Err(Refusal::Absent {
            what: "the MTP pending-hidden state",
        });
    }

    let dst = out.all("out_width(0)")?;
    let hidden_size = dst.width;

    ctx.fire(
        Fire::at(
            "attn/attention_naive.cuh",
            crate::jit::symbol(&format!("::pie::attn::mtp_shift_hidden<{}>", T::CPP)),
        )
        .apply(Launch::per_row(
            dst.rows.unsigned_abs(),
            attention_naive::BLOCK,
        )),
        &[
            target_hidden.arg(),
            pending_hidden.arg(),
            qo_indptr.arg(),
            slot_ids.arg(),
            out.arg(),
            num_requests.arg(),
            hidden_size.arg(),
        ],
    )
}

#[routine(bf16, whole)]
pub fn mtp_update_pending_hidden<T>(
    ctx: &Ctx<'_>,
    target_hidden: In<Tensor<T>>,
    qo_indptr: In<Tensor<i32>>,
    rsv: In<Struct<RecurrentState>>,
    pending: In<Struct<MtpPendingHidden>>,
) -> Result<(), Refusal>
where
    *const T: Abi + kernels::Bind<crate::jit::ArgValue>,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    // The request count is the CSR operand's own row count.
    let num_requests = qo_indptr.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_ids = rsv.slots;
    let pending_hidden = pending.ptr.cast_mut().cast::<T>();
    if pending_hidden.is_null() {
        return Err(Refusal::Absent {
            what: "the MTP pending-hidden state",
        });
    }

    let src = target_hidden.all("in_width(0)")?;
    let hidden_size = src.width;

    ctx.fire(
        Fire::at(
            "attn/attention_naive.cuh",
            crate::jit::symbol(&format!(
                "::pie::attn::mtp_update_pending_hidden<{}>",
                T::CPP
            )),
        )
        .apply(Launch::per_row(
            num_requests.unsigned_abs(),
            attention_naive::BLOCK,
        )),
        &[
            target_hidden.arg(),
            pending_hidden.arg(),
            qo_indptr.arg(),
            slot_ids.arg(),
            num_requests.arg(),
            hidden_size.arg(),
        ],
    )
}

#[allow(clippy::similar_names)]
#[routine(whole, untraced)]
pub fn mla_prepare_bf16(
    ctx: &Ctx<'_>,
    layer: MlaLayer,
    kv_a: In<Tensor<bf16>>,
    kv_a_norm_weight: Const<Tensor<bf16>>,
    q_b: In<Tensor<bf16>>,
    kv_c: Out<Tensor<bf16>>,
    k_pe: Out<Tensor<bf16>>,
    q_nope: Out<Tensor<bf16>>,
    q_pe: Out<Tensor<bf16>>,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: Const<f32>,
    theta: Const<f32>,
    interleaved: bool,
    kv_a_row_stride: i32,
    yarn: Option<Yarn>,
    qo_indptr: In<Tensor<i32>>,
    positions: In<Tensor<i32>>,
    kvc: In<Struct<KvCache>>,
    row_valid: In<Tensor<i32>>,
    num_requests: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let qo_indptr = qo_indptr.ptr as *const u32;
    let positions = positions.ptr;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let row_valid = row_valid.ptr as *const u8;
    let num_requests = *num_requests;

    #[must_use]
    pub fn mla_q_blocks(heads: i32, heads_per_block: i32) -> i32 {
        if heads_per_block <= 0 {
            return 0;
        }
        heads.saturating_add(heads_per_block - 1) / heads_per_block
    }

    #[must_use]
    pub fn mla_heads_per_block(rope: i32) -> i32 {
        let half = rope / 2;
        if half >= MLA_PREPARE_BLOCK {
            1
        } else if half > 0 {
            MLA_PREPARE_BLOCK / half
        } else {
            1
        }
    }

    pub const MLA_PREPARE_BLOCK: i32 = 256;

    let kv_lora = layer.kv_lora_rank;
    let rope = layer.qk_rope_head_dim;
    let stride = if kv_a_row_stride > 0 {
        kv_a_row_stride
    } else {
        kv_lora + rope
    };
    let per_block = mla_heads_per_block(rope);
    let blocks = mla_q_blocks(heads, per_block);

    let (low_dim, high_dim) = match yarn {
        Some(y) => crate::rope::ramp_bounds(
            rope,
            *theta,
            y.beta_fast,
            y.beta_slow,
            y.original_max_position,
        ),
        None => (0.0, 0.0),
    };
    let yarn_factor = yarn.map_or(-1.0_f32, |y| y.factor);
    let yarn_mscale = yarn.map_or(1.0_f32, |y| y.attention_factor);

    ctx.fire(
        Fire::at(
            "attn/mla_paged.cuh",
            "::pie::attn::mla_prepare<::pie::i32(256)>",
        )
        .apply(Launch::grid(
            [
                kv_c.rows.unsigned_abs(),
                blocks.saturating_add(1).max(1).unsigned_abs(),
                1,
            ],
            [MLA_PREPARE_BLOCK.unsigned_abs(), 1, 1],
        )),
        &[
            kv_a.arg(),
            kv_a_norm_weight.arg(),
            q_b.arg(),
            kv_c.arg(),
            k_pe.arg(),
            q_nope.arg(),
            q_pe.arg(),
            layer.ckv_pages.cast::<bf16>().arg(),
            layer.kpe_pages.cast::<bf16>().arg(),
            positions.arg(),
            qo_indptr.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            row_valid.arg(),
            num_requests.arg(),
            layer.page_size.arg(),
            heads.arg(),
            kv_lora.arg(),
            qk_nope_head_dim.arg(),
            rope.arg(),
            stride.arg(),
            eps.arg(),
            theta.arg(),
            interleaved.arg(),
            per_block.arg(),
            yarn_factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            yarn_mscale.arg(),
        ],
    )
}

#[routine(whole, untraced)]
pub fn write_mla_to_pages(
    ctx: &Ctx<'_>,
    layer: MlaLayer,
    ckv_curr: In<Tensor<bf16>>,
    kpe_curr: In<Tensor<bf16>>,
    qo_indptr: In<Tensor<i32>>,
    kvc: In<Struct<KvCache>>,
    row_valid: In<Tensor<i32>>,
    num_requests: Const<i32>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;

    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let row_valid = row_valid.ptr as *const u8;
    let num_requests = *num_requests;

    pub const MLA_WRITE_BLOCK: u32 = 256;

    ctx.fire(
        Fire::at("attn/mla_paged.cuh", "::pie::attn::write_mla").apply(Launch::per_row(
            ckv_curr.rows.unsigned_abs(),
            MLA_WRITE_BLOCK,
        )),
        &[
            ckv_curr.arg(),
            kpe_curr.arg(),
            layer.ckv_pages.cast::<bf16>().arg(),
            layer.kpe_pages.cast::<bf16>().arg(),
            qo_indptr.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            row_valid.arg(),
            num_requests.arg(),
            layer.page_size.arg(),
            layer.kv_lora_rank.arg(),
            layer.qk_rope_head_dim.arg(),
        ],
    )
}

const DSV4_META_BLOCK: u32 = 128;

#[routine(out(out_pos = like(positions)), out(out_req = like(positions)), out(out_rope = like(positions)))]
pub fn dsv4_boundary_meta_decode(
    ctx: &Ctx<'_>,
    positions: In<Tensor<i32>>,
    out_pos: Out<Tensor<i32>>,
    out_req: Out<Tensor<i32>>,
    out_rope: Out<Tensor<i32>>,
    ratio: Const<i32>,
    row_valid: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let ratio = *ratio;
    let row_valid = row_valid.ptr as *const u8;
    if ratio <= 0 {
        return Err(Refusal::Narrow {
            what: "ratio",
            at: i64::from(ratio),
        });
    }

    ctx.fire(
        Fire::at(
            "attn/dsv4_compress.cuh",
            "::pie::attn::dsv4_boundary_meta_decode<::pie::i32>",
        )
        .apply(Launch::flat(out_pos.rows.unsigned_abs(), DSV4_META_BLOCK)),
        &[
            positions.arg(),
            out_pos.arg(),
            out_req.arg(),
            out_rope.arg(),
            out_pos.rows.arg(),
            ratio.arg(),
            row_valid.arg(),
        ],
    )
}

#[routine(whole, out(out_pos = like(positions)), out(out_req = like(positions)), out(out_rope = like(positions)))]
pub fn dsv4_boundary_meta_paged(
    ctx: &Ctx<'_>,
    positions: In<Tensor<i32>>,
    out_pos: Out<Tensor<i32>>,
    out_req: Out<Tensor<i32>>,
    out_rope: Out<Tensor<i32>>,
    ratio: Const<i32>,
    row_valid: In<Tensor<i32>>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let ratio = *ratio;
    let row_valid = row_valid.ptr as *const u8;
    // The request count is the CSR operand's own row count.
    let num_requests = qo_indptr.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    if ratio <= 0 {
        return Err(Refusal::Narrow {
            what: "ratio",
            at: i64::from(ratio),
        });
    }

    ctx.fire(
        Fire::at(
            "attn/dsv4_compress.cuh",
            "::pie::attn::dsv4_boundary_meta_paged<::pie::i32>",
        )
        .apply(Launch::flat(out_pos.rows.unsigned_abs(), DSV4_META_BLOCK)),
        &[
            positions.arg(),
            qo_indptr.arg(),
            out_pos.arg(),
            out_req.arg(),
            out_rope.arg(),
            out_pos.rows.arg(),
            num_requests.arg(),
            ratio.arg(),
            row_valid.arg(),
        ],
    )
}

#[routine(whole)]
pub fn attention_compressed_paged_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    lse_out: Out<Tensor<f32>>,
    ratio: Const<i32>,
    num_q_heads: Const<i32>,
    head_dim: Const<i32>,
    kvc: In<Struct<KvCache>>,
    sm_scale: Const<f32>,
    positions: In<Tensor<i32>>,
    request_of_token: In<Tensor<i32>>,
    comp_kv: In<Struct<Dsv4CompKvPages>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let ratio = *ratio;
    let num_q_heads = *num_q_heads;
    let head_dim = *head_dim;
    let page_size = kvc.page_size;
    let sm_scale = *sm_scale;

    let positions = positions.ptr;
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let req_of_token = request_of_token.ptr;
    let comp_kv_pages = comp_kv.ptr;

    const DSV4_ATTN_BLOCK: u32 = 128;

    let smem = head_dim
        .max(0)
        .unsigned_abs()
        .saturating_add(DSV4_ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));

    ctx.fire(
        Fire::at(
            "attn/dsv4_compress.cuh",
            "::pie::attn::compressed_attn_paged",
        )
        .apply(
            Launch::grid(
                [o.rows.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                [DSV4_ATTN_BLOCK, 1, 1],
            )
            .smem(smem),
        ),
        &[
            q.arg(),
            comp_kv_pages.arg(),
            o.arg(),
            lse_out.arg(),
            positions.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            req_of_token.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            ratio.arg(),
            page_size.arg(),
            sm_scale.arg(),
        ],
    )
}

#[routine(bf16, out(idx_k = like(idx_k)))]
pub fn dsa_index_knorm_rope<T>(
    ctx: &Ctx<'_>,
    idx_k: InOut<Tensor<T>>,
    k_norm_weight: Const<Tensor<T>>,
    k_norm_bias: Const<Tensor<T>>,
    rope_dim: Const<i32>,
    theta: Const<f32>,
    eps: Const<f32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal>
where
    *const T: Abi + kernels::Bind<crate::jit::ArgValue>,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
{
    let rope_dim = *rope_dim;
    let theta = *theta;
    let eps = *eps;

    let positions = positions.ptr;

    let dst = idx_k.all("out_width(0)")?;
    let head_dim = dst.width;

    ctx.fire(
        Fire::at(
            "attn/dsa_indexer.cuh",
            crate::jit::symbol(&format!("::pie::attn::index_knorm_rope<{}>", T::CPP)),
        )
        .apply(Launch::per_row(
            dst.rows.unsigned_abs(),
            dsa_indexer::K_BLOCK,
        )),
        &[
            idx_k.arg(),
            k_norm_weight.arg(),
            k_norm_bias.arg(),
            positions.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
            eps.arg(),
        ],
    )
}

#[routine(bf16, out(idx_q = split(idx_q, head_dim)))]
pub fn dsa_index_q_rope<T>(
    ctx: &Ctx<'_>,
    idx_q: InOut<Tensor<T>>,
    n_heads: Const<i32>,
    head_dim: Const<i32>,
    rope_dim: Const<i32>,
    theta: Const<f32>,
    positions: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let n_heads = *n_heads;
    let head_dim = *head_dim;
    let rope_dim = *rope_dim;
    let theta = *theta;

    let positions = positions.ptr;

    ctx.fire(
        Fire::at(
            "attn/dsa_indexer.cuh",
            crate::jit::symbol(&format!("::pie::attn::index_q_rope<{}>", T::CPP)),
        )
        .apply(Launch::per_row(
            idx_q.rows.unsigned_abs(),
            dsa_indexer::q_rope_block(n_heads),
        )),
        &[
            idx_q.arg(),
            positions.arg(),
            n_heads.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
        ],
    )
}

#[routine(whole)]
pub fn dsa_index_topk_mask(
    ctx: &Ctx<'_>,
    idx_q: In<Tensor<bf16>>,
    idx_k: In<Tensor<bf16>>,
    idx_w: In<Tensor<bf16>>,
    mask: Out<Tensor<u8>>,
    n_heads: Const<i32>,
    head_dim: Const<i32>,
    topk: Const<i32>,
) -> Result<(), Refusal> {
    let smem = mask
        .rows
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));

    ctx.fire(
        Fire::at(
            "attn/dsa_indexer.cuh",
            "::pie::attn::index_topk_mask<::pie::bf16>",
        )
        .apply(Launch::per_row(mask.rows.unsigned_abs(), dsa_indexer::K_BLOCK).smem(smem)),
        &[
            idx_q.arg(),
            idx_k.arg(),
            idx_w.arg(),
            mask.arg(),
            mask.rows.arg(),
            n_heads.arg(),
            head_dim.arg(),
            topk.arg(),
        ],
    )
}

pub mod mla_params {
    use super::bf16;
    use crate::by_value;
    use crate::jit::{ByValue, Layout};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct UintFastdiv {
        pub opaque: [u64; 3],
    }

    impl UintFastdiv {
        #[must_use]
        pub const fn new(d: u32) -> Self {
            let d64 = d as u64;
            let magic = if d == 0 {
                0
            } else {
                let q = u64::MAX / d64;
                let r = u64::MAX % d64;
                q + if r + 1 == d64 { 1 } else { 0 } + 1
            };
            Self {
                opaque: [d64, magic, d64],
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct MlaParams {
        pub q_nope: *mut bf16,
        pub q_pe: *mut bf16,
        pub ckv: *mut bf16,
        pub kpe: *mut bf16,
        pub partial_o: *mut bf16,
        pub partial_lse: *mut f32,
        pub final_o: *mut bf16,
        pub final_lse: *mut f32,
        pub q_indptr: *mut i32,
        pub kv_indptr: *mut i32,
        pub partial_indptr: *mut i32,
        pub merge_packed_offset_start: *mut i32,
        pub merge_packed_offset_end: *mut i32,
        pub merge_partial_packed_offset_start: *mut i32,
        pub merge_partial_packed_offset_end: *mut i32,
        pub merge_partial_stride: *mut i32,
        pub kv_indices: *mut i32,
        pub q_len: *mut i32,
        pub kv_len: *mut i32,
        pub q_start: *mut i32,
        pub kv_start: *mut i32,
        pub kv_end: *mut i32,
        pub work_indptr: *mut i32,
        pub block_size: UintFastdiv,
        pub num_heads: UintFastdiv,
        pub q_nope_stride_n: u32,
        pub q_nope_stride_h: u32,
        pub q_pe_stride_n: u32,
        pub q_pe_stride_h: u32,
        pub ckv_stride_page: u32,
        pub ckv_stride_n: u32,
        pub kpe_stride_page: u32,
        pub kpe_stride_n: u32,
        pub o_stride_n: u32,
        pub o_stride_h: u32,
        pub sm_scale: f32,
        pub ckv_scale: f32,
        pub kpe_scale: f32,
        pub return_lse_base_on_e: bool,
    }

    by_value! {
        MlaParams as "::flashinfer::MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>",
        untagged,
        probe = "nvrtc-probes/attn_mla_params.py",
        size = 288, align = 8,
        {
            q_nope               @ 0   as "q_nope",
            work_indptr          @ 176 as "work_indptr",
            block_size           @ 184 as "block_size",
            num_heads            @ 208 as "num_heads",
            q_nope_stride_n      @ 232 as "q_nope_stride_n",
            o_stride_h           @ 268 as "o_stride_h",
            sm_scale             @ 272 as "sm_scale",
            ckv_scale            @ 276 as "ckv_scale",
            kpe_scale            @ 280 as "kpe_scale",
            return_lse_base_on_e @ 284 as "return_lse_base_on_e",
        }
    }

    pub static LAYOUTS: &[Layout] = &[<MlaParams as ByValue>::LAYOUT];

    const _: () = assert!(
        ::core::mem::size_of::<UintFastdiv>() == 24,
        "UintFastdiv: sizeof disagrees with the measured ::flashinfer::uint_fastdiv \
         (24, NOT 4 — see nvrtc-probes/attn_mla_params.py)",
    );
    const _: () = assert!(
        ::core::mem::align_of::<UintFastdiv>() == 8,
        "UintFastdiv: alignof disagrees with the measured ::flashinfer::uint_fastdiv",
    );
}

pub mod mla_naive {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};

    use kernels::{Bind, Fire};

    pub const NAIVE_BLOCK: u32 = 256;

    pub const NAIVE_WARPS: i32 = NAIVE_BLOCK as i32 / 32;

    pub const NAIVE_MAX_PER: i32 = 16;

    pub const NAIVE_MAX_PE_PER: i32 = 4;

    pub const WAVE_TARGET: i64 = 296;

    pub const FORCED_GROUP: i32 = 0;

    pub const MMA_BM: i32 = 16;

    pub const MMA_SMEM_BYTES: u32 = 100_032;

    #[must_use]
    pub const fn naive_smem_bytes(kv_lora_rank: i32) -> u32 {
        let per = NAIVE_WARPS as i64 * kv_lora_rank as i64 + 2 * NAIVE_WARPS as i64;
        let bytes = per * 4;
        if bytes < 0 { 0 } else { bytes as u32 }
    }

    #[must_use]
    pub enum MlaNaive {
        LaunchedScalar,
        LaunchedMma,
        Declined(NaiveDecline),
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum NaiveDecline {
        NoTokens,
        MissingIndptr,
        UnsupportedKvLoraRank,
        UnsupportedRopeDim,
    }

    #[must_use]
    pub const fn mma_supported(kv_lora_rank: i32, qk_rope_head_dim: i32, num_heads: i32) -> bool {
        kv_lora_rank == 512 && qk_rope_head_dim == 64 && num_heads % MMA_BM == 0
    }

    #[must_use]
    pub fn head_group(num_heads: i32, total_tokens: i32) -> i32 {
        let mut g = NAIVE_WARPS;
        while g > 1
            && (num_heads % g != 0
                || i64::from(total_tokens) * i64::from(num_heads / g) < WAVE_TARGET)
        {
            g >>= 1;
        }
        g
    }

    #[must_use]
    pub fn head_group_forced(num_heads: i32, forced: i32) -> i32 {
        let mut g = forced;
        while g > 1 && (num_heads % g != 0 || NAIVE_WARPS % g != 0) {
            g >>= 1;
        }
        g
    }

    #[derive(Clone, Copy, Debug)]
    pub struct NaiveShape {
        pub kv_lora_rank: i32,
        pub qk_rope_head_dim: i32,
        pub page_size: i32,
        pub total_tokens: i32,
        pub num_requests: i32,
        pub num_heads: i32,
        pub sm_scale: f32,
        pub causal: bool,
        pub index_mask_stride: i32,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct NaivePtrs {
        pub q_nope: *const bf16,
        pub q_pe: *const bf16,
        pub ckv_pages: *const bf16,
        pub kpe_pages: *const bf16,
        pub qo_indptr: *const u32,
        pub kv_page_indices: *const u32,
        pub kv_page_indptr: *const u32,
        pub kv_last_page_lens: *const u32,
        pub o: *mut bf16,
        pub index_mask: *const u8,
    }

    #[must_use]
    pub enum NaivePlan {
        Scalar { launch: Launch, head_group: i32 },
        Mma { launch: Launch },
        Declined(NaiveDecline),
    }

    pub fn plan(shape: NaiveShape, have_indptr: bool) -> NaivePlan {
        pub const MMA_THREADS: u32 = 256;

        if shape.total_tokens <= 0 {
            return NaivePlan::Declined(NaiveDecline::NoTokens);
        }

        if !have_indptr {
            return NaivePlan::Declined(NaiveDecline::MissingIndptr);
        }

        if mma_supported(shape.kv_lora_rank, shape.qk_rope_head_dim, shape.num_heads) {
            #[allow(clippy::cast_sign_loss)]
            let launch = Launch::grid(
                [
                    (shape.num_heads / MMA_BM).max(0) as u32,
                    shape.total_tokens.max(0) as u32,
                    1,
                ],
                [MMA_THREADS, 1, 1],
            )
            .smem(MMA_SMEM_BYTES);
            return NaivePlan::Mma { launch };
        }

        let ckv = shape.kv_lora_rank;
        let kpe = shape.qk_rope_head_dim;
        if ckv % 32 != 0 || ckv / 32 > NAIVE_MAX_PER {
            return NaivePlan::Declined(NaiveDecline::UnsupportedKvLoraRank);
        }
        if kpe % 32 != 0 || kpe / 32 > NAIVE_MAX_PE_PER {
            return NaivePlan::Declined(NaiveDecline::UnsupportedRopeDim);
        }

        let g = if FORCED_GROUP > 0 {
            head_group_forced(shape.num_heads, FORCED_GROUP)
        } else {
            head_group(shape.num_heads, shape.total_tokens)
        };
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch::grid(
            [
                shape.total_tokens.max(0) as u32,
                (shape.num_heads / g.max(1)).max(1) as u32,
                1,
            ],
            [NAIVE_BLOCK, 1, 1],
        )
        .smem(naive_smem_bytes(ckv));
        NaivePlan::Scalar {
            launch,
            head_group: g,
        }
    }

    pub fn fire(ctx: &Ctx<'_>, ptrs: NaivePtrs, shape: NaiveShape) -> Result<MlaNaive, Refusal> {
        let have_indptr = !ptrs.qo_indptr.is_null()
            && !ptrs.kv_page_indptr.is_null()
            && !ptrs.kv_last_page_lens.is_null();
        match plan(shape, have_indptr) {
            NaivePlan::Declined(why) => Ok(MlaNaive::Declined(why)),
            NaivePlan::Mma { launch } => {
                ctx.fire(
                    Fire::at(
                        "attn/attention_mla_naive.cuh",
                        "::pie::attn::mla_naive::mma_detail::mla_mma_paged_kernel",
                    )
                    .apply(launch),
                    &[
                        ptrs.q_nope.arg(),
                        ptrs.q_pe.arg(),
                        ptrs.ckv_pages.arg(),
                        ptrs.kpe_pages.arg(),
                        ptrs.qo_indptr.arg(),
                        ptrs.kv_page_indices.arg(),
                        ptrs.kv_page_indptr.arg(),
                        ptrs.kv_last_page_lens.arg(),
                        ptrs.o.arg(),
                        ptrs.index_mask.arg(),
                        shape.index_mask_stride.arg(),
                        shape.num_requests.arg(),
                        shape.num_heads.arg(),
                        shape.page_size.arg(),
                        shape.sm_scale.arg(),
                        shape.causal.arg(),
                    ],
                )?;
                Ok(MlaNaive::LaunchedMma)
            }
            NaivePlan::Scalar { launch, head_group } => {
                ctx.fire(
                    Fire::at(
                        "attn/attention_mla_naive.cuh",
                        "::pie::attn::mla_naive::mla_naive_paged_kernel",
                    )
                    .apply(launch),
                    &[
                        ptrs.q_nope.arg(),
                        ptrs.q_pe.arg(),
                        ptrs.ckv_pages.arg(),
                        ptrs.kpe_pages.arg(),
                        ptrs.qo_indptr.arg(),
                        ptrs.kv_page_indices.arg(),
                        ptrs.kv_page_indptr.arg(),
                        ptrs.kv_last_page_lens.arg(),
                        ptrs.o.arg(),
                        ptrs.index_mask.arg(),
                        shape.index_mask_stride.arg(),
                        shape.num_requests.arg(),
                        shape.num_heads.arg(),
                        shape.kv_lora_rank.arg(),
                        shape.qk_rope_head_dim.arg(),
                        shape.page_size.arg(),
                        shape.sm_scale.arg(),
                        shape.causal.arg(),
                        head_group.arg(),
                    ],
                )?;
                Ok(MlaNaive::LaunchedScalar)
            }
        }
    }

    pub const NAIVE_OPT_IN_BYTES_UNREACHED: u32 = 200 * 1024;
}

pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use super::{Ctx, Refusal};
    use crate::attn::plan::MlaPlanInfo;
    use crate::jit::Abi;
    use crate::jit::{Launch, Root};

    use kernels::Fire;

    pub static ROOT: Root = Root::new("attn/attention_mla_fa2.cuh");

    pub mod inst {
        pub const MLA: [[&str; 2]; 3] = [
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
            [
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
                 ::pie::attn::mla_fa2::Params>",
                "::flashinfer::mla::BatchMLAPagedAttentionKernel\
                 <::pie::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
                 ::pie::attn::mla_fa2::Params>",
            ],
        ];
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Arm {
        pub stages: u32,
        pub cta_tile_kv: u32,
        pub qk_shard: bool,
        pub smem: u32,
    }

    pub const ARMS: [Arm; 3] = [
        Arm {
            stages: 2,
            cta_tile_kv: 64,
            qk_shard: true,
            smem: 221_696,
        },
        Arm {
            stages: 2,
            cta_tile_kv: 32,
            qk_shard: true,
            smem: 147_968,
        },
        Arm {
            stages: 1,
            cta_tile_kv: 16,
            qk_shard: false,
            smem: 92_672,
        },
    ];

    #[must_use]
    pub const fn arm_index(smem_limit_per_sm: u32) -> Option<usize> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[must_use]
    pub const fn arm_for(smem_limit_per_sm: u32) -> Option<Arm> {
        match arm_index(smem_limit_per_sm) {
            Some(i) => Some(ARMS[i]),
            None => None,
        }
    }

    #[must_use]
    pub const fn options() -> &'static [&'static str] {
        ROOT.options
    }

    pub const SYMBOLS: [[&str; 2]; 3] = [
        ["attn::mla_fa2_kv64_full", "attn::mla_fa2_kv64_causal"],
        ["attn::mla_fa2_kv32_full", "attn::mla_fa2_kv32_causal"],
        ["attn::mla_fa2_kv16_full", "attn::mla_fa2_kv16_causal"],
    ];

    pub const SMEM_ECHO: [&str; 3] = [
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 2u, true, 64u>>",
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 2u, true, 32u>>",
        "&::pie::attn::mla_fa2::smem_bytes_mla<::pie::attn::mla_fa2::Traits<true, 1u, false, 16u>>",
    ];

    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        pub page_size: u32,
        pub num_heads: u32,
        pub kv_lora_rank: u32,
        pub qk_rope_head_dim: u32,
        pub sm_scale: f32,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Buffers {
        pub int_buffer: *mut u8,
        pub float_buffer: *mut u8,
        pub q_nope: *mut bf16,
        pub q_pe: *mut bf16,
        pub ckv_pages: *mut bf16,
        pub kpe_pages: *mut bf16,
        pub out: *mut bf16,
        pub kv_page_indices: *mut i32,
        pub lse: *mut f32,
    }

    /// The MLA parameter block a launch is handed, laid out from a plan.
    ///
    /// # Safety
    ///
    /// `buffers` must hold device addresses into the arena the plan was
    /// measured against: `int_buffer` must reach every offset `plan` states,
    /// and the float and output pointers must address `shape`'s extents. The
    /// result borrows none of them -- it is a block of raw addresses -- so they
    /// must outlive the launch that reads it, not this call.
    #[must_use]
    pub unsafe fn pack(
        plan: &MlaPlanInfo,
        shape: Shape,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
            unsafe { base.cast::<T>().offset(offset as isize) }
        }

        let int_buf = buffers.int_buffer;
        let float_buf = buffers.float_buffer;
        MlaParams {
            q_nope: buffers.q_nope,
            q_pe: buffers.q_pe,
            ckv: buffers.ckv_pages,
            kpe: buffers.kpe_pages,
            partial_o: unsafe { offset_ptr(float_buf, plan.partial_o_offset) },
            partial_lse: unsafe { offset_ptr(float_buf, plan.partial_lse_offset) },
            final_o: buffers.out,
            final_lse: if want_lse {
                buffers.lse
            } else {
                ::core::ptr::null_mut()
            },
            q_indptr: unsafe { offset_ptr(int_buf, plan.q_indptr_offset) },
            kv_indptr: unsafe { offset_ptr(int_buf, plan.kv_indptr_offset) },
            partial_indptr: unsafe { offset_ptr(int_buf, plan.partial_indptr_offset) },
            merge_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_start_offset)
            },
            merge_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_end_offset)
            },
            merge_partial_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_start_offset)
            },
            merge_partial_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_end_offset)
            },
            merge_partial_stride: unsafe { offset_ptr(int_buf, plan.merge_partial_stride_offset) },
            kv_indices: buffers.kv_page_indices,
            q_len: unsafe { offset_ptr(int_buf, plan.q_len_offset) },
            kv_len: unsafe { offset_ptr(int_buf, plan.kv_len_offset) },
            q_start: unsafe { offset_ptr(int_buf, plan.q_start_offset) },
            kv_start: unsafe { offset_ptr(int_buf, plan.kv_start_offset) },
            kv_end: unsafe { offset_ptr(int_buf, plan.kv_end_offset) },
            work_indptr: unsafe { offset_ptr(int_buf, plan.work_indptr_offset) },
            block_size: UintFastdiv::new(shape.page_size),
            num_heads: UintFastdiv::new(shape.num_heads),
            q_nope_stride_n: shape.num_heads * shape.kv_lora_rank,
            q_nope_stride_h: shape.kv_lora_rank,
            q_pe_stride_n: shape.num_heads * shape.qk_rope_head_dim,
            q_pe_stride_h: shape.qk_rope_head_dim,
            ckv_stride_page: shape.page_size * shape.kv_lora_rank,
            ckv_stride_n: shape.kv_lora_rank,
            kpe_stride_page: shape.page_size * shape.qk_rope_head_dim,
            kpe_stride_n: shape.qk_rope_head_dim,
            o_stride_n: shape.num_heads * shape.kv_lora_rank,
            o_stride_h: shape.kv_lora_rank,
            sm_scale: shape.sm_scale,
            ckv_scale: 1.0,
            kpe_scale: 1.0,
            return_lse_base_on_e: true,
        }
    }

    #[must_use]
    pub const fn grid(plan: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch::grid(
            [plan.num_blks_x as u32, plan.num_blks_y as u32, 1],
            [256, 1, 1],
        )
        .smem(arm.smem)
        .cooperative()
    }

    pub fn fire(
        ctx: &Ctx<'_>,
        arm: usize,
        causal: bool,
        params: &MlaParams,
        launch: Launch,
    ) -> Result<(), Refusal> {
        let Some(row) = inst::MLA.get(arm) else {
            return Err(Refusal::Absent {
                what: "a `DISPATCH_SMEM_CONFIG` arm for this device",
            });
        };

        ctx.fire(
            Fire::at("attn/attention_mla_fa2.cuh", row[usize::from(causal)]).apply(launch),
            &[params.arg()],
        )
    }
}

#[must_use]
pub enum MlaDispatch {
    Fa2 { arm: usize },
    Naive(mla_naive::MlaNaive),
}

/// # Safety
///
/// `ctx`'s stream must be live, and `plan` must have been measured against
/// the same page table the `Ctx` will answer for -- the page indices, the
/// indptrs and the last-page lengths are read as device addresses without
/// a bound check, because the plan is what bounded them.
#[routine(untraced)]
pub unsafe fn dispatch_attention_mla_bf16(
    ctx: &Ctx<'_>,
    plan: &MlaPlan,
    q_nope: In<Tensor<bf16>>,
    q_pe: In<Tensor<bf16>>,
    layer: MlaLayer,
    o: Out<Tensor<bf16>>,
    index_mask_stride: i32,
    num_heads: i32,
    sm_scale: Const<f32>,
    causal: bool,
    kvc: In<Struct<KvCache>>,
    qo_indptr: In<Tensor<i32>>,
    maskv: In<Struct<AttnMask>>,
    num_requests: Const<i32>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<MlaDispatch, Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    if maskv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the mask view this statement names",
        });
    }
    let maskv = unsafe { &*maskv.ptr };
    let kv_page_indices = kvc.page_indices as *const u32;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;

    let lse = lse.map_or(core::ptr::null_mut(), |l| l.ptr);
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let index_mask = maskv.mask;
    let num_requests = *num_requests;
    let Some(major) = ctx.compute_capability_major() else {
        return Err(Refusal::Device {
            why: "the device would not say its compute capability, which is the whole \
                  of this dispatch: FA2 MLA writes zeros on sm_100",
        });
    };

    if major >= 10 {
        let ptrs = mla_naive::NaivePtrs {
            q_nope: q_nope.ptr,
            q_pe: q_pe.ptr,
            ckv_pages: layer.ckv_pages.cast::<bf16>().cast_const(),
            kpe_pages: layer.kpe_pages.cast::<bf16>().cast_const(),
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            o: o.ptr,
            index_mask,
        };
        let shape = mla_naive::NaiveShape {
            kv_lora_rank: layer.kv_lora_rank,
            qk_rope_head_dim: layer.qk_rope_head_dim,
            page_size: layer.page_size,
            total_tokens: o.rows,
            num_requests,
            num_heads,
            sm_scale: *sm_scale,
            causal,
            index_mask_stride,
        };
        return mla_naive::fire(ctx, ptrs, shape).map(MlaDispatch::Naive);
    }

    let Some(arm) = mla_fa2::arm_index(fa2::plan::fa_device().max_smem_per_sm) else {
        return Err(Refusal::Absent {
            what: "a `DISPATCH_SMEM_CONFIG` arm for this device's shared memory per SM",
        });
    };
    let shape = mla_fa2::Shape {
        page_size: layer.page_size.unsigned_abs(),
        num_heads: num_heads.unsigned_abs(),
        kv_lora_rank: layer.kv_lora_rank.unsigned_abs(),
        qk_rope_head_dim: layer.qk_rope_head_dim.unsigned_abs(),
        sm_scale: *sm_scale,
    };
    let buffers = mla_fa2::Buffers {
        int_buffer: plan.int_arena.cast::<u8>(),
        float_buffer: plan.float_arena.cast::<u8>(),
        q_nope: q_nope.ptr.cast_mut(),
        q_pe: q_pe.ptr.cast_mut(),
        ckv_pages: layer.ckv_pages.cast::<bf16>(),
        kpe_pages: layer.kpe_pages.cast::<bf16>(),
        out: o.ptr,
        kv_page_indices: (kv_page_indices).cast::<i32>().cast_mut(),
        lse,
    };

    let params = unsafe { mla_fa2::pack(&plan.info, shape, buffers, !lse.is_null()) };

    mla_fa2::fire(
        ctx,
        arm,
        causal,
        &params,
        mla_fa2::grid(&plan.info, mla_fa2::ARMS[arm]),
    )?;
    Ok(MlaDispatch::Fa2 { arm })
}

pub mod qkv_fused {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};

    use crate::jit::abi::Tensor;
    use crate::views::KvCache;

    use kernels::raises::Struct;
    use kernels::routine::{Const, In, Out};
    use kernels::{Bind, Fire};
    use kernels_macros::routine;

    #[routine]
    pub fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        ctx: &Ctx<'_>,
        packed: In<Tensor<bf16>>,
        q_out: Out<Tensor<bf16>>,
        q_weight: Const<Tensor<bf16>>,
        k_weight: Const<Tensor<bf16>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
        kvc: In<Struct<KvCache>>,
        theta: Const<f32>,
        eps: Const<f32>,
        positions: In<Tensor<i32>>,
        row_valid: In<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let page_size = kvc.page_size;
        let hnd_layout = kvc.layout != 0;
        let theta = *theta;
        let eps = *eps;

        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let positions = positions.ptr;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let kv_last_page_lens = kvc.last_page_lens as *const u32;
        let row_valid = row_valid.ptr as *const u8;

        pub const PACKED_BLOCK: u32 = 256;

        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let num_q_heads = q_out.all("out_width(0)")?.width / head_dim;
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();

        ctx.fire(
            Fire::at(
                "attn/qkv_fused.cuh",
                "::pie::attn::qkv_packed_qk_norm_rope_vnorm_write_kv<::pie::i32(256)>",
            )
            .apply(Launch::grid(
                [packed.rows.unsigned_abs(), heads, 1],
                [PACKED_BLOCK, 1, 1],
            )),
            &[
                packed.arg(),
                q_out.arg(),
                k_pages.arg(),
                v_pages.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                row_valid.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }

    fn warp_instantiation(head_dim: i32, rope_table: bool) -> Option<&'static str> {
        Some(match (head_dim, rope_table) {
            (64, true) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(64), true>"
            }
            (64, false) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(64), false>"
            }
            (128, true) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(128), true>"
            }
            (128, false) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(128), false>"
            }
            (256, true) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(256), true>"
            }
            (256, false) => {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv_warp<::pie::i32(256), false>"
            }
            _ => return None,
        })
    }

    #[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
    pub fn qkv_decode_fused_dispatch(
        ctx: &Ctx<'_>,
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        rope_table: *const f32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        w_page: *const u32,
        w_off: *const u32,
        row_valid: *const u8,
        win: *const u32,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
    ) -> Result<(), Refusal> {
        pub const WARP_BLOCK: u32 = 256;

        const fn block_instantiation(rope_table: bool) -> &'static str {
            if rope_table {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv<::pie::i32(128), true>"
            } else {
                "::pie::attn::qkv_decode_qk_norm_rope_write_kv<::pie::i32(128), false>"
            }
        }

        const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

        pub const DECODE_BLOCK: u32 = 128;

        if q_out.is_null() {
            return Err(Refusal::Absent { what: "q_out" });
        }

        let use_rope_table = !rope_table.is_null();
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();

        if let Some(instantiation) = warp_instantiation(head_dim, use_rope_table) {
            let units = num_requests.unsigned_abs().saturating_mul(heads);

            return ctx.fire(
                Fire::at("attn/qkv_fused.cuh", instantiation).apply(Launch::grid(
                    [units.div_ceil(WARPS_PER_BLOCK), 1, 1],
                    [WARP_BLOCK, 1, 1],
                )),
                &[
                    packed.arg(),
                    q_out.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    q_weight.arg(),
                    k_weight.arg(),
                    positions.arg(),
                    rope_table.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    row_valid.arg(),
                    win.arg(),
                    num_requests.arg(),
                    num_q_heads.arg(),
                    num_kv_heads.arg(),
                    page_size.arg(),
                    hnd_layout.arg(),
                    theta.arg(),
                    eps.arg(),
                ],
            );
        }
        ctx.fire(
            Fire::at("attn/qkv_fused.cuh", block_instantiation(use_rope_table)).apply(
                Launch::grid(
                    [num_requests.unsigned_abs(), heads, 1],
                    [DECODE_BLOCK, 1, 1],
                ),
            ),
            &[
                packed.arg(),
                q_out.arg(),
                k_pages.arg(),
                v_pages.arg(),
                q_weight.arg(),
                k_weight.arg(),
                positions.arg(),
                rope_table.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                w_page.arg(),
                w_off.arg(),
                row_valid.arg(),
                win.arg(),
                num_q_heads.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }

    const QKV_DECODE_FUSED_DISPATCH_ROW: ::kernels::routine::Routine<crate::Plane> =
        ::kernels::untraced!(
            crate::Plane,
            "qkv_decode_fused_dispatch",
            qkv_decode_fused_dispatch,
            namespace = "attn"
        )
        .internal();

    #[cfg(not(target_family = "wasm"))]
    #[::linkme::distributed_slice(crate::ROUTINES)]
    #[allow(non_upper_case_globals)]
    static QKV_DECODE_FUSED_DISPATCH_ROUTINE: ::kernels::routine::Routine<crate::Plane> =
        QKV_DECODE_FUSED_DISPATCH_ROW;

    #[cfg(target_family = "wasm")]
    ::inventory::submit! { crate::Registered(QKV_DECODE_FUSED_DISPATCH_ROW) }

    #[allow(clippy::fn_params_excessive_bools)]
    #[routine]
    pub fn qkv_decode_qk_norm_rope_write_kv_bf16(
        ctx: &Ctx<'_>,
        packed: In<Tensor<bf16>>,
        q_out: Out<Tensor<bf16>>,
        q_weight: Const<Tensor<bf16>>,
        k_weight: Const<Tensor<bf16>>,
        // NULLABLE: the launcher dispatches on a null table (the tableless
        // instantiation), and the statement may omit the operand.
        rope_table: Option<In<Tensor<f32>>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
        kvc: In<Struct<KvCache>>,
        theta: Const<f32>,
        eps: Const<f32>,
        positions: In<Tensor<i32>>,
        row_valid: In<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let page_size = kvc.page_size;
        let hnd_layout = kvc.layout != 0;
        let theta = *theta;
        let eps = *eps;

        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let positions = positions.ptr;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let kv_last_page_lens = kvc.last_page_lens as *const u32;
        let w_page = kvc.write_page as *const u32;
        let w_off = kvc.write_offset as *const u32;
        let row_valid = row_valid.ptr as *const u8;

        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let packed_width = packed.all("in_width(0)")?.width;
        let num_q_heads = (packed_width - 2 * num_kv_heads * head_dim) / head_dim;
        qkv_decode_fused_dispatch(
            ctx,
            packed.ptr,
            q_out.ptr,
            k_pages.cast::<bf16>(),
            v_pages.cast::<bf16>(),
            q_weight.v,
            k_weight.v,
            positions,
            rope_table.map_or(core::ptr::null(), |t| t.ptr),
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            w_page,
            w_off,
            row_valid,
            core::ptr::null(),
            packed.rows,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            hnd_layout,
            theta,
            eps,
        )
    }
}

pub mod dsv4_compress {
    use super::bf16;
    use super::{Ctx, Launch, Refusal};
    use crate::jit::abi::Tensor;
    use crate::views::{Dsv4Ape, Dsv4CompKvPages, Dsv4StateKv, Dsv4StateScore, KvCache};

    use kernels::raises::Struct;
    use kernels::routine::{Const, In, Out};
    use kernels::{Bind, Fire};
    use kernels_macros::routine;

    #[expect(
        clippy::cast_sign_loss,
        reason = "both are guarded positive by every caller"
    )]
    fn route_rows(rows: i32, width: i32) -> Launch {
        let (rows, width) = (rows as u32, width as u32);
        Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
    }

    #[routine]
    pub fn dsv4_compress_gather_paged_bf16(
        ctx: &Ctx<'_>,
        boundary_pos: In<Tensor<i32>>,
        boundary_req: In<Tensor<i32>>,
        out: Out<Tensor<bf16>>,
        ratio: Const<i32>,
        coff: Const<i32>,
        kvc: In<Struct<KvCache>>,
        state_kv: In<Struct<Dsv4StateKv>>,
        state_score: In<Struct<Dsv4StateScore>>,
        ape: In<Struct<Dsv4Ape>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let num_entries = boundary_pos.rows;
        let ratio = *ratio;
        let coff = *coff;
        let page_size = kvc.page_size;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let state_kv = state_kv.ptr;
        let state_score = state_score.ptr;
        let ape = ape.ptr;
        let head_dim = out.all("out_width(0)")?.width;

        ctx.fire(
            Fire::at(
                "attn/dsv4_compress.cuh",
                "::pie::attn::dsv4_compress_gather_paged<::pie::bf16>",
            )
            .apply(route_rows(num_entries, head_dim)),
            &[
                state_kv.arg(),
                state_score.arg(),
                ape.arg(),
                boundary_pos.arg(),
                boundary_req.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                out.arg(),
                head_dim.arg(),
                ratio.arg(),
                coff.arg(),
                page_size.arg(),
            ],
        )
    }

    #[routine(whole)]
    pub fn dsv4_store_comp_entries_bf16(
        ctx: &Ctx<'_>,
        entries: In<Tensor<bf16>>,
        boundary_pos: In<Tensor<i32>>,
        boundary_req: In<Tensor<i32>>,
        kvc: In<Struct<KvCache>>,
        comp_kv: In<Struct<Dsv4CompKvPages>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let num_entries = entries.rows;
        let page_size = kvc.page_size;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let comp_kv_pages = comp_kv.ptr.cast_mut();

        let head_dim = entries.all("in_width(0)")?.width;

        ctx.fire(
            Fire::at(
                "attn/dsv4_compress.cuh",
                "::pie::attn::dsv4_store_comp_entries<::pie::bf16>",
            )
            .apply(route_rows(num_entries, head_dim)),
            &[
                entries.arg(),
                comp_kv_pages.arg(),
                boundary_pos.arg(),
                boundary_req.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                head_dim.arg(),
                page_size.arg(),
            ],
        )
    }
}

pub mod kv_paged {
    use crate::attn::{KvDType, KvScheme, kv_dtype, kv_scheme};

    use crate::jit::abi::MaybeConst;
    use crate::jit::fp8_kind;

    use super::bf16;
    use super::{Ctx, Launch, Refusal, scheme_byte};
    use crate::jit::abi::Tensor;
    use crate::views::KvCache;
    use core::ffi::c_void;

    use kernels::raises::Struct;
    use kernels::routine::{Const, In, Out};
    use kernels::{Bind, Fire};
    use kernels_macros::routine;

    const BLOCK: u32 = 256;

    fn fp8_kind_of(storage_dtype: kv_dtype) -> fp8_kind {
        const NV_E5M2: u32 = 1;

        const NV_E4M3: u32 = 0;

        fp8_kind(if storage_dtype == kv_dtype::of(KvDType::Fp8E5M2) {
            NV_E5M2
        } else {
            NV_E4M3
        })
    }

    fn fp4_block_size(block_size: i32) -> i32 {
        if block_size > 0 { block_size } else { 16 }
    }

    #[must_use]
    pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
        if page_size <= 0 {
            return 0;
        }
        (total_tokens + page_size - 1) / page_size + num_requests
    }

    #[routine]
    pub fn write_kv_explicit_bf16(
        ctx: &Ctx<'_>,
        k_curr: In<Tensor<bf16>>,
        v_curr: In<Tensor<bf16>>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
        row_valid: In<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let hnd = kvc.layout != 0;
        let has_envelopes = kvc.has_envelopes;
        let is_native_bf16 = kvc.native_bf16;

        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let w_page = kvc.write_page as *const u32;
        let w_off = kvc.write_offset as *const u32;
        let row_valid = row_valid.ptr as *const u8;
        let k_env_min = kvc.env_min;
        let k_env_max = kvc.env_max;
        assert!(
            is_native_bf16,
            "attn::write_kv_explicit_bf16 requires native bf16 KV cache"
        );

        let instantiation = if hnd {
            "::pie::attn::write_kv_explicit<\
                                ::pie::true_type::value>"
        } else {
            "::pie::attn::write_kv_explicit<::pie::false_type::value>"
        };

        ctx.fire(
            Fire::at("attn/kv_paged.cuh", instantiation)
                .apply(Launch::per_row(k_curr.rows.unsigned_abs(), BLOCK)),
            &[
                k_curr.arg(),
                v_curr.arg(),
                k_pages.arg(),
                v_pages.arg(),
                w_page.arg(),
                w_off.arg(),
                MaybeConst::new(row_valid).arg(),
                k_curr.rows.arg(),
                page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        )?;

        if has_envelopes && !hnd {
            let _ = crate::layout::envelope_merge_written(
                ctx,
                In {
                    ptr: k_curr.ptr,
                    rows: k_curr.rows,
                    width: head_dim,
                },
                In {
                    ptr: w_page,
                    rows: k_curr.rows,
                    width: 1,
                },
                In {
                    ptr: w_off,
                    rows: k_curr.rows,
                    width: 1,
                },
                MaybeConst::new(row_valid),
                Out {
                    ptr: (k_env_min).cast::<bf16>().cast_mut(),
                    rows: k_curr.rows,
                    width: head_dim,
                },
                Out {
                    ptr: (k_env_max).cast::<bf16>().cast_mut(),
                    rows: k_curr.rows,
                    width: head_dim,
                },
                k_curr.rows,
                num_kv_heads,
                head_dim,
            );
        }
        Ok(())
    }

    #[routine(whole)]
    pub fn write_kv_explicit_bf16_devwin(
        ctx: &Ctx<'_>,
        k_curr: In<Tensor<bf16>>,
        v_curr: In<Tensor<bf16>>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
        row_valid: In<Tensor<i32>>,
        n_max: Const<i32>,
        win_start: Const<i32>,
        win_len: Const<i32>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let hnd = kvc.layout != 0;
        let has_envelopes = kvc.has_envelopes;
        let is_native_bf16 = kvc.native_bf16;

        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let w_page = kvc.write_page as *const u32;
        let w_off = kvc.write_offset as *const u32;
        let win_d = crate::stage_peel_window(ctx, "attn::write_kv_devwin", *win_start, *win_len)?;
        let row_valid = row_valid.ptr as *const u8;
        let n_max = *n_max;
        assert!(
            is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        assert!(
            !has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let instantiation = if hnd {
            "::pie::attn::write_kv_explicit_devwin<::pie::true_type::value>"
        } else {
            "::pie::attn::write_kv_explicit_devwin<::pie::false_type::value>"
        };

        ctx.fire(
            Fire::at("attn/kv_paged.cuh", instantiation)
                .apply(Launch::per_row((n_max).unsigned_abs(), BLOCK)),
            &[
                k_curr.arg(),
                v_curr.arg(),
                k_pages.arg(),
                v_pages.arg(),
                w_page.arg(),
                w_off.arg(),
                MaybeConst::new(row_valid).arg(),
                win_d.arg(),
                n_max.arg(),
                page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        )
    }

    #[routine]
    pub fn write_kv_to_pages_bf16(
        ctx: &Ctx<'_>,
        k_curr: In<Tensor<bf16>>,
        v_curr: In<Tensor<bf16>>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,
        // THE TWO LEGS AGREE BY PREFIX, which is why `first_token` precedes
        // the CSR here even though the body reads the CSR first.
        // `attn::write_kv_to_pages` is a declaration standing for a CHOICE:
        // a model text states the outer name and `Boot::route` picks this
        // body or `write_kv_to_pages_quantised` from a fact the CHECKPOINT
        // settles, long after the trace was recorded. One statement therefore
        // has to bind correctly under either leg, and a `Source` is
        // POSITIONAL -- so the only arrangement that can work is the one
        // where the shorter leg's operand list is a prefix of the longer's.
        // The quantised appender takes no row-validity mask (it refuses a
        // non-zero write origin outright and has no partial rows to skip), so
        // `row_valid` is what hangs off the end.
        head_dim: Const<i32>,
        first_token: In<Tensor<i32>>,
        qo_indptr: In<Tensor<i32>>,
        row_valid: In<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let hnd = kvc.layout != 0;
        let has_envelopes = kvc.has_envelopes;
        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        // The request count is the CSR operand's own row count.
        let num_requests = qo_indptr.rows;
        let qo_indptr = qo_indptr.ptr as *const u32;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let kv_last_page_lens = kvc.last_page_lens as *const u32;
        let row_valid = row_valid.ptr as *const u8;
        let k_env_min = kvc.env_min;
        let k_env_max = kvc.env_max;
        let first_token = first_token.ptr as i32;
        let launch_tokens = k_curr.rows - first_token;

        let instantiation = if hnd {
            "::pie::attn::write_kv<\
                                                ::pie::true_type::value>"
        } else {
            "::pie::attn::write_kv<::pie::false_type::value>"
        };

        ctx.fire(
            Fire::at("attn/kv_paged.cuh", instantiation)
                .apply(Launch::per_row(launch_tokens.unsigned_abs(), BLOCK)),
            &[
                k_curr.arg(),
                v_curr.arg(),
                k_pages.arg(),
                v_pages.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                MaybeConst::new(row_valid).arg(),
                MaybeConst::<u32>::none().arg(),
                num_requests.arg(),
                page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
                first_token.arg(),
            ],
        )?;

        if has_envelopes && !hnd && k_curr.rows > 0 {
            let _ = crate::layout::envelope_update_appended(
                ctx,
                In {
                    ptr: k_pages.cast::<bf16>().cast_const(),
                    rows: k_curr.rows,
                    width: head_dim,
                },
                In {
                    ptr: qo_indptr,
                    rows: num_requests,
                    width: 1,
                },
                In {
                    ptr: kv_page_indices,
                    rows: num_requests,
                    width: 1,
                },
                In {
                    ptr: kv_page_indptr,
                    rows: num_requests,
                    width: 1,
                },
                In {
                    ptr: kv_last_page_lens,
                    rows: num_requests,
                    width: 1,
                },
                Out {
                    ptr: (k_env_min).cast::<bf16>().cast_mut(),
                    rows: k_curr.rows,
                    width: head_dim,
                },
                Out {
                    ptr: (k_env_max).cast::<bf16>().cast_mut(),
                    rows: k_curr.rows,
                    width: head_dim,
                },
                num_requests,
                max_touched_pages(k_curr.rows, num_requests, page_size),
                page_size,
                num_kv_heads,
                head_dim,
            );
        }
        Ok(())
    }

    #[routine]
    pub fn write_kv_to_pages_quantised(
        ctx: &Ctx<'_>,
        k_curr: In<Tensor<bf16>>,
        v_curr: In<Tensor<bf16>>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
        first_token: In<Tensor<i32>>,
        qo_indptr: In<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let block_size = kvc.block_size;
        let scheme = kvc.scheme_byte;
        let storage_dtype = kvc.storage_dtype;
        let first_token = first_token.ptr as i32;
        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let k_scales = kvc.key_scales as *mut core::ffi::c_void;
        let v_scales = kvc.value_scales as *mut core::ffi::c_void;
        // The request count is the CSR operand's own row count.
        let num_requests = qo_indptr.rows;
        let qo_indptr = qo_indptr.ptr as *const u32;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let kv_last_page_lens = kvc.last_page_lens as *const u32;
        if first_token != 0 {
            return Err(Refusal::Absent {
                what: "a quantised appender that skips the first tokens",
            });
        }
        let scheme = kv_scheme(scheme_byte(scheme));
        let storage_dtype = kv_dtype(scheme_byte(storage_dtype));
        let h_kv = num_kv_heads;
        let d = head_dim;
        let tokens = k_curr.rows.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match scheme.scheme() {
            Some(KvScheme::Fp8PerTensor) => ctx.fire(
                Fire::at("attn/kv_paged.cuh", "::pie::attn::write_kv_fp8_per_tensor")
                    .apply(Launch::per_row(tokens, BLOCK)),
                &[
                    k_curr.arg(),
                    v_curr.arg(),
                    k_pages.arg(),
                    v_pages.arg(),
                    qo_indptr.arg(),
                    kv_page_indices.arg(),
                    kv_page_indptr.arg(),
                    kv_last_page_lens.arg(),
                    num_requests.arg(),
                    page_size.arg(),
                    h_kv.arg(),
                    d.arg(),
                    fp8_kind_of(storage_dtype).arg(),
                ],
            ),
            Some(KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead) => {
                let instantiation = if scheme == kv_scheme::of(KvScheme::Fp8PerTokenHead) {
                    "::pie::attn::write_kv_per_token_head<::pie::true_type::value>"
                } else {
                    "::pie::attn::write_kv_per_token_head<::pie::false_type::value>"
                };

                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                ctx.fire(
                    Fire::at("attn/kv_paged.cuh", instantiation)
                        .apply(Launch::grid([tokens, heads, 1], [BLOCK, 1, 1]).smem(smem)),
                    &[
                        k_curr.arg(),
                        v_curr.arg(),
                        k_pages.arg(),
                        v_pages.arg(),
                        k_scales.cast::<f32>().arg(),
                        v_scales.cast::<f32>().arg(),
                        qo_indptr.arg(),
                        kv_page_indices.arg(),
                        kv_page_indptr.arg(),
                        kv_last_page_lens.arg(),
                        num_requests.arg(),
                        page_size.arg(),
                        h_kv.arg(),
                        d.arg(),
                    ],
                )
            }

            Some(KvScheme::Fp4Block) => {
                let block_size = fp4_block_size(block_size);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                ctx.fire(
                    Fire::at("attn/kv_paged.cuh", "::pie::attn::write_kv_fp4_block").apply(
                        Launch::grid([tokens, heads, blocks.unsigned_abs()], [32, 1, 1]),
                    ),
                    &[
                        k_curr.arg(),
                        v_curr.arg(),
                        k_pages.arg(),
                        v_pages.arg(),
                        k_scales.cast::<f32>().arg(),
                        v_scales.cast::<f32>().arg(),
                        qo_indptr.arg(),
                        kv_page_indices.arg(),
                        kv_page_indptr.arg(),
                        kv_last_page_lens.arg(),
                        num_requests.arg(),
                        page_size.arg(),
                        h_kv.arg(),
                        d.arg(),
                        block_size.arg(),
                    ],
                )
            }

            Some(KvScheme::Native) => Err(Refusal::Absent {
                what: "a quantised writer for Native storage",
            }),

            None => Err(Refusal::Absent {
                what: "a KV scheme this byte names",
            }),
        }
    }

    #[must_use]
    pub const fn write_kv_to_pages(is_native_bf16: bool) -> &'static str {
        if is_native_bf16 {
            concat!("attn::", stringify!(write_kv_to_pages_bf16))
        } else {
            concat!("attn::", stringify!(write_kv_to_pages_quantised))
        }
    }

    #[allow(dead_code)]
    fn the_map_names_two_real_fns() {
        let _ = (write_kv_to_pages_bf16, write_kv_to_pages_quantised);
    }

    const WRITE_KV_TO_PAGES_ROW: ::kernels::routine::Routine<crate::Plane> = ::kernels::untraced!(
        crate::Plane,
        "write_kv_to_pages",
        write_kv_to_pages_bf16,
        namespace = "attn"
    )
    // The KV-append role: `Kv::append` resolves `canon = kv_append` and
    // this declared name is what a text states; `Boot::route` still
    // picks the bf16/quantised body.
    .canon("kv_append");

    #[cfg(not(target_family = "wasm"))]
    #[::linkme::distributed_slice(crate::ROUTINES)]
    #[allow(non_upper_case_globals)]
    static WRITE_KV_TO_PAGES_ROUTINE: ::kernels::routine::Routine<crate::Plane> =
        WRITE_KV_TO_PAGES_ROW;

    #[cfg(target_family = "wasm")]
    ::inventory::submit! { crate::Registered(WRITE_KV_TO_PAGES_ROW) }

    #[allow(clippy::too_many_arguments)]
    pub fn dequant_fp8_per_tensor_pages_active(
        ctx: &Ctx<'_>,
        k_pages: *mut u8,
        v_pages: *mut u8,
        k_bf16_pages: *mut c_void,
        v_bf16_pages: *mut c_void,
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scheme: kv_scheme,
        storage_dtype: kv_dtype,
        is_native_bf16: bool,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
    ) -> Result<(), Refusal> {
        if is_native_bf16 {
            return Err(Refusal::Absent {
                what: "quantised pages on a bf16 layer",
            });
        }
        if scheme != kv_scheme::of(KvScheme::Fp8PerTensor) {
            return Err(Refusal::Absent {
                what: "an fp8-per-tensor layer",
            });
        }

        let (logical_n, page_elems, launch) =
            active_geometry(page_size, num_kv_heads, head_dim, num_pages_in_batch);

        ctx.fire(
            Fire::at("attn/kv_paged.cuh", "::pie::attn::dequant_fp8_pages_active").apply(launch),
            &[
                k_pages.cast::<u8>().cast_const().arg(),
                v_pages.cast::<u8>().cast_const().arg(),
                k_bf16_pages.cast::<bf16>().arg(),
                v_bf16_pages.cast::<bf16>().arg(),
                kv_page_indices.arg(),
                logical_n.arg(),
                page_elems.arg(),
                fp8_kind_of(storage_dtype).arg(),
            ],
        )
    }

    fn active_geometry(
        page_size: i32,
        num_kv_heads: i32,
        head_dim: i32,
        num_pages_in_batch: i32,
    ) -> (i64, i32, Launch) {
        let page_elems = page_size * num_kv_heads * head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        (
            logical_n,
            page_elems,
            Launch::grid([blocks as u32, 1, 1], [BLOCK, 1, 1]),
        )
    }

    #[routine]
    #[allow(clippy::too_many_arguments)]
    pub fn dequant_kv_cache_layer_to_bf16_active(
        ctx: &Ctx<'_>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,
        head_dim: Const<i32>,
    ) -> Result<(), Refusal> {
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;
        let num_kv_heads = *num_kv_heads;
        let head_dim = *head_dim;
        let block_size = kvc.block_size;
        let scheme = kvc.scheme_byte;
        let storage_dtype = kvc.storage_dtype;
        let is_native_bf16 = kvc.native_bf16;

        let k_pages = kvc.keys;
        let v_pages = kvc.values;
        let k_scales = kvc.key_scales as *mut core::ffi::c_void;
        let v_scales = kvc.value_scales as *mut core::ffi::c_void;
        let k_bf16_pages = kvc.bf16_keys as *mut core::ffi::c_void;
        let v_bf16_pages = kvc.bf16_values as *mut core::ffi::c_void;
        let kv_page_indices = kvc.page_indices as *const u32;
        let num_pages_in_batch = kvc.pages_in_batch;
        if is_native_bf16 {
            return Ok(());
        }
        let scheme = kv_scheme(scheme_byte(scheme));
        let storage_dtype = kv_dtype(scheme_byte(storage_dtype));
        let (logical_n, _page_elems, launch) =
            active_geometry(page_size, num_kv_heads, head_dim, num_pages_in_batch);

        match scheme.scheme() {
            Some(KvScheme::Fp8PerTensor) => dequant_fp8_per_tensor_pages_active(
                ctx,
                k_pages,
                v_pages,
                k_bf16_pages,
                v_bf16_pages,
                page_size,
                num_kv_heads,
                head_dim,
                scheme,
                storage_dtype,
                is_native_bf16,
                kv_page_indices,
                num_pages_in_batch,
            ),
            Some(KvScheme::Fp8PerTokenHead) => ctx.fire(
                Fire::at(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp8_per_token_head_pages_active<::pie::bf16>",
                )
                .apply(launch),
                &[
                    (k_pages).cast::<u8>().cast_const().arg(),
                    (v_pages).cast::<u8>().cast_const().arg(),
                    (k_scales).cast::<f32>().cast_const().arg(),
                    (v_scales).cast::<f32>().cast_const().arg(),
                    (k_bf16_pages).cast::<bf16>().arg(),
                    (v_bf16_pages).cast::<bf16>().arg(),
                    (kv_page_indices).arg(),
                    logical_n.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ],
            ),
            Some(KvScheme::Int8PerTokenHead) => ctx.fire(
                Fire::at(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_int8_per_token_head_pages_active<::pie::bf16>",
                )
                .apply(launch),
                &[
                    (k_pages).cast::<i8>().cast_const().arg(),
                    (v_pages).cast::<i8>().cast_const().arg(),
                    (k_scales).cast::<f32>().cast_const().arg(),
                    (v_scales).cast::<f32>().cast_const().arg(),
                    (k_bf16_pages).cast::<bf16>().arg(),
                    (v_bf16_pages).cast::<bf16>().arg(),
                    (kv_page_indices).arg(),
                    logical_n.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ],
            ),
            Some(KvScheme::Fp4Block) => ctx.fire(
                Fire::at(
                    "attn/kv_paged.cuh",
                    "::pie::attn::dequant_fp4_pages_active<::pie::bf16>",
                )
                .apply(launch),
                &[
                    (k_pages).cast::<u8>().cast_const().arg(),
                    (v_pages).cast::<u8>().cast_const().arg(),
                    (k_scales).cast::<f32>().cast_const().arg(),
                    (v_scales).cast::<f32>().cast_const().arg(),
                    (k_bf16_pages).cast::<bf16>().arg(),
                    (v_bf16_pages).cast::<bf16>().arg(),
                    (kv_page_indices).arg(),
                    logical_n.arg(),
                    page_size.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                    fp4_block_size(block_size).arg(),
                ],
            ),
            Some(KvScheme::Native) => Err(Refusal::Absent {
                what: "a quantised dequant for Native storage",
            }),

            None => Err(Refusal::Absent {
                what: "a KV scheme this byte names",
            }),
        }
    }
}

const BLOCK: u32 = 256;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    #[must_use]
    const fn head_dim_block(head_dim: u32) -> u32 {
        const SINK_BLOCK_MAX: u32 = 128;

        const SINK_BLOCK_MIN: u32 = 32;

        if head_dim < SINK_BLOCK_MIN {
            SINK_BLOCK_MIN
        } else if head_dim > SINK_BLOCK_MAX {
            SINK_BLOCK_MAX
        } else {
            head_dim
        }
    }

    Launch::grid([rows, heads, 1], [head_dim_block(head_dim), 1, 1])
}

#[must_use]
const fn per_head(rows: u32, heads: u32) -> Launch {
    const PAD_BLOCK: u32 = 128;

    Launch::grid([heads, rows, 1], [PAD_BLOCK, 1, 1])
}

#[routine(out(lse = like(lse)))]
pub fn lse_log2_to_ln(ctx: &Ctx<'_>, lse: InOut<Tensor<f32>>) -> Result<(), Refusal> {
    let elems = lse.all("out_width(0)")?.elements();
    let Ok(elems) = u32::try_from(elems) else {
        return Err(Refusal::Empty {
            what: "lse elements",
        });
    };
    let n = elems as usize;

    ctx.fire(
        Fire::at(
            "attn/attn_sink.cuh",
            "::pie::attn::lse_log2_to_ln<::pie::attn::f32>",
        )
        .apply(elementwise(elems)),
        &[lse.arg(), n.arg()],
    )
}

#[routine(bf16, out(o = like(o)))]
pub fn attention_sink_rescale<T>(
    ctx: &Ctx<'_>,
    o: InOut<Tensor<T>>,
    lse: In<Tensor<f32>>,
    sinks: Const<Tensor<T>>,
    num_q_heads: Const<i32>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    let num_q_heads = *num_q_heads;
    let head_dim = *head_dim;

    ctx.fire(
        Fire::at(
            "attn/attn_sink.cuh",
            crate::jit::symbol(&format!("::pie::attn::attn_sink_rescale<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            o.rows.unsigned_abs(),
            num_q_heads.unsigned_abs(),
            head_dim.unsigned_abs(),
        )),
        &[
            o.arg(),
            lse.arg(),
            sinks.arg(),
            o.rows.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
        ],
    )
}

#[routine]
pub fn split_qkv_bf16_devwin(
    ctx: &Ctx<'_>,
    packed: In<Tensor<bf16>>,
    q_out: Out<Tensor<bf16>>,
    k_out: Out<Tensor<bf16>>,
    v_out: Out<Tensor<bf16>>,
    n_max: Const<i32>,
    win_start: Const<i32>,
    win_len: Const<i32>,
) -> Result<(), Refusal> {
    let win = crate::stage_peel_window(ctx, "attn::split_qkv_devwin", *win_start, *win_len)?;
    let n_max = *n_max;

    pub const SPLIT_BLOCK: u32 = 256;

    let (q_dim, kv_dim) = (
        q_out.all("out_width(0)")?.width,
        k_out.all("out_width(1)")?.width,
    );
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);

    ctx.fire(
        Fire::at(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv_devwin<::pie::bf16>",
        )
        .apply(Launch::grid(
            [xblocks.max(1), (n_max).unsigned_abs(), 1],
            [SPLIT_BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q_out.arg(),
            k_out.arg(),
            v_out.arg(),
            win.arg(),
            q_dim.arg(),
            kv_dim.arg(),
        ],
    )
}

#[routine(whole)]
pub fn attention_naive_paged(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    kvc: In<Struct<KvCache>>,
    head_dim: Const<i32>,
    num_kv_heads: Const<i32>,
    window_left: Const<i32>,
    sm_scale: Const<f32>,
    logits_soft_cap: Const<f32>,
    qo_indptr: In<Tensor<i32>>,
    lse_out: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let page_size = kvc.page_size;
    let head_dim = *head_dim;
    let num_kv_heads = *num_kv_heads;
    let scheme = kvc.scheme_byte;
    let storage_dtype = kvc.storage_dtype;
    let block_size = kvc.block_size;
    let window_left = *window_left;
    let sm_scale = *sm_scale;
    let logits_soft_cap = *logits_soft_cap;
    // The request count is the CSR operand's own row count.
    let num_requests = qo_indptr.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let kv_page_indices = kvc.page_indices as *const u32;
    let k_pages = kvc.keys;
    let v_pages = kvc.values;
    let k_scales = kvc.key_scales as *mut core::ffi::c_void;
    let v_scales = kvc.value_scales as *mut core::ffi::c_void;
    let kv_page_indptr = kvc.page_indptr as *const u32;
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
    let lse_out = lse_out.map_or(core::ptr::null_mut(), |l| l.ptr);

    pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

    pub const PAGED_BLOCK: u32 = 128;

    if head_dim > PAGED_MAX_HEAD_DIM {
        return Err(Refusal::Wide {
            what: "head_dim",
            at: i64::from(head_dim),
            max: i64::from(PAGED_MAX_HEAD_DIM),
        });
    }
    let src = q.all("in_width(0)")?;
    let num_q_heads = src.width.checked_div(head_dim).unwrap_or(0);
    let smem = ((head_dim).unsigned_abs() + PAGED_BLOCK) * 4;

    ctx.fire(
        Fire::at(
            "attn/attention_naive_paged.cuh",
            "::pie::attn::naive_paged_attn<::pie::i32(128)>",
        )
        .apply(
            Launch::grid(
                [
                    num_requests.unsigned_abs(),
                    src.rows.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                ],
                [PAGED_BLOCK, 1, 1],
            )
            .smem(smem),
        ),
        &[
            q.arg(),
            (k_pages).cast_const().arg(),
            (v_pages).cast_const().arg(),
            (k_scales).cast::<f32>().cast_const().arg(),
            (v_scales).cast::<f32>().cast_const().arg(),
            o.arg(),
            qo_indptr.arg(),
            kv_page_indices.arg(),
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            core::ptr::null::<u8>().arg(),
            core::ptr::null::<i32>().arg(),
            num_q_heads.arg(),
            num_kv_heads.arg(),
            head_dim.arg(),
            page_size.arg(),
            kv_scheme(scheme_byte(scheme)).arg(),
            kv_dtype(scheme_byte(storage_dtype)).arg(),
            block_size.arg(),
            window_left.arg(),
            sm_scale.arg(),
            logits_soft_cap.arg(),
            lse_out.arg(),
        ],
    )
}

#[routine(bf16, out(out = like(prefix)))]
pub fn attn_res_blend<T>(
    ctx: &Ctx<'_>,
    prefix: In<Tensor<T>>,
    blocks: In<Tensor<T>>,
    // WEIGHTS, so the statement names them and the chain binds them --
    // an `In` slot here read operands the text never places.
    norm_weight: Const<Tensor<T>>,
    proj_weight: Const<Tensor<T>>,
    out: Out<Tensor<T>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    let dst = out.all("out_width(0)")?;
    let h = dst.width;

    let b = blocks.all("in_width(1)")?.width / h;

    ctx.fire(
        Fire::at(
            "attn/attn_res.cuh",
            crate::jit::symbol(&format!("::pie::attn::attn_res_blend<{}>", T::CPP)),
        )
        .apply(Launch::per_row(dst.rows.unsigned_abs(), BLOCK)),
        &[
            prefix.arg(),
            blocks.arg(),
            norm_weight.arg(),
            proj_weight.arg(),
            out.arg(),
            b.arg(),
            h.arg(),
            dst.rows.arg(),
            eps.arg(),
        ],
    )
}

#[routine(bf16)]
pub fn pad_head_dim<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    padded: Out<Tensor<T>>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    let head_dim = *head_dim;

    let num_heads = packed.width.checked_div(head_dim).unwrap_or(0);
    let head_dim_padded = padded.width.checked_div(num_heads).unwrap_or(0);
    if let Some(why) = head_dim_refusal(packed.rows, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }

    ctx.fire(
        Fire::at(
            "attn/head_dim_pad.cuh",
            crate::jit::symbol(&format!("::pie::attn::pad_head_dim<{}>", T::CPP)),
        )
        .apply(per_head(
            packed.rows.unsigned_abs(),
            num_heads.unsigned_abs(),
        )),
        &[
            packed.arg(),
            padded.arg(),
            num_heads.arg(),
            head_dim.arg(),
            head_dim_padded.arg(),
        ],
    )
}

#[routine(bf16, out(packed = rows(padded) x const(head_dim)))]
pub fn strip_head_dim<T>(
    ctx: &Ctx<'_>,
    padded: In<Tensor<T>>,
    packed: Out<Tensor<T>>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    let head_dim = *head_dim;

    let num_heads = packed.width.checked_div(head_dim).unwrap_or(0);
    let head_dim_padded = padded.width.checked_div(num_heads).unwrap_or(0);
    if let Some(why) = head_dim_refusal(padded.rows, num_heads, head_dim, head_dim_padded) {
        return Err(why);
    }

    ctx.fire(
        Fire::at(
            "attn/head_dim_pad.cuh",
            crate::jit::symbol(&format!("::pie::attn::strip_head_dim<{}>", T::CPP)),
        )
        .apply(per_head(
            padded.rows.unsigned_abs(),
            num_heads.unsigned_abs(),
        )),
        &[
            padded.arg(),
            packed.arg(),
            num_heads.arg(),
            head_dim.arg(),
            head_dim_padded.arg(),
        ],
    )
}

#[must_use]
fn head_dim_refusal(
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Option<Refusal> {
    if num_tokens <= 0 {
        return Some(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Some(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Some(Refusal::Empty { what: "head_dim" });
    }
    if head_dim_padded < head_dim {
        return Some(Refusal::Narrow {
            what: "head_dim_padded",
            at: i64::from(head_dim_padded),
        });
    }
    None
}

fn softcap_elems<P: Copy>(x: &kernels::Region<P>) -> Result<usize, Refusal> {
    let elems = x.elements();
    usize::try_from(elems).map_err(|_| Refusal::Narrow {
        what: "logit elements",
        at: i64::from(elems),
    })
}

fn softcap_launch(cap: f32, n: usize) -> Result<Launch, Refusal> {
    if cap.is_nan() || cap <= 0.0 {
        return Err(Refusal::Unstated {
            what: "a logit soft cap",
        });
    }
    let Ok(elems) = u32::try_from(n) else {
        return Err(Refusal::Wide {
            what: "logit elements",
            at: i64::from(i32::MAX),
            max: i64::from(i32::MAX),
        });
    };
    Ok(elementwise(elems))
}

#[routine(bf16)]
pub fn logit_softcap<T>(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<T>>,
    cap: Const<f32>,
) -> Result<(), Refusal> {
    let cap = *cap;
    let n = softcap_elems(&x.all("out_width(0)")?)?;
    let launch = softcap_launch(cap, n)?;

    ctx.fire(
        Fire::at(
            "attn/softcap.cuh",
            crate::jit::symbol(&format!("::pie::attn::logit_softcap<{}>", T::CPP)),
        )
        .apply(launch),
        &[x.arg(), cap.arg(), n.arg()],
    )
}

#[routine(internal)]
pub fn logit_softcap_f16(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<f16>>,
    cap: Const<f32>,
) -> Result<(), Refusal> {
    let cap = *cap;

    let n = softcap_elems(&x.all("out_width(0)")?)?;
    let launch = softcap_launch(cap, n)?;
    ctx.fire(
        Fire::at("attn/softcap.cuh", "::pie::attn::logit_softcap<::pie::f16>").apply(launch),
        &[x.arg(), cap.arg(), n.arg()],
    )
}

#[routine(bf16)]
pub fn kimi_split_q_b<T>(
    ctx: &Ctx<'_>,
    q_b: In<Tensor<T>>,
    q_nope: Out<Tensor<T>>,
    q_pe: Out<Tensor<T>>,
    heads: Const<i32>,
    nope: Const<i32>,
    rope: Const<i32>,
) -> Result<(), Refusal> {
    let width = i64::from(*heads) * (i64::from(*nope) + i64::from(*rope));
    let total = i64::from(q_b.rows) * width;
    if total > i64::from(i32::MAX) {
        return Err(Refusal::Wide {
            what: "rows",
            at: i64::from(q_b.rows),
            max: i64::from(i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX)),
        });
    }
    let total = total as i32;

    ctx.fire(
        Fire::at(
            "attn/kimi_mla.cuh",
            crate::jit::symbol(&format!("::pie::attn::split_q_b<{}>", T::CPP)),
        )
        .apply(elementwise(total.unsigned_abs())),
        &[
            q_b.arg(),
            q_nope.arg(),
            q_pe.arg(),
            total.arg(),
            heads.arg(),
            nope.arg(),
            rope.arg(),
        ],
    )
}

#[routine(bf16)]
pub fn kimi_split_kv_a_norm<T>(
    ctx: &Ctx<'_>,
    kv_a: In<Tensor<T>>,
    norm_weight: Const<Tensor<T>>,
    kv_c: Out<Tensor<T>>,
    k_pe: Out<Tensor<T>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    #[must_use]
    const fn rms(rows: u32) -> Launch {
        Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
    }

    let (kv_lora, rope, src) = (
        kv_c.all("out_width(0)")?.width,
        k_pe.all("out_width(1)")?.width,
        kv_a.all("in_width(0)")?,
    );
    let src_row_stride = src.stride;
    if *src_row_stride < kv_lora + rope {
        return Err(Refusal::Narrow {
            what: "src_row_stride",
            at: i64::from(*src_row_stride),
        });
    }

    ctx.fire(
        Fire::at(
            "attn/kimi_mla.cuh",
            crate::jit::symbol(&format!("::pie::attn::split_kv_a_norm<{}, 256>", T::CPP)),
        )
        .apply(rms(src.rows.unsigned_abs())),
        &[
            kv_a.arg(),
            norm_weight.arg(),
            kv_c.arg(),
            k_pe.arg(),
            kv_lora.arg(),
            rope.arg(),
            src_row_stride.arg(),
            eps.arg(),
        ],
    )
}

#[routine(bf16, out(o_out = like(o1)), out(lse_out = like(lse1)))]
pub fn combine_attn_outputs<T>(
    ctx: &Ctx<'_>,
    o1: In<Tensor<T>>,
    lse1: In<Tensor<f32>>,
    o2: In<Tensor<T>>,
    lse2: In<Tensor<f32>>,
    o_out: Out<Tensor<T>>,
    lse_out: Out<Tensor<f32>>,
    num_heads: Const<i32>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    #[must_use]
    const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
        #[must_use]
        const fn combine_block(head_dim: u32) -> u32 {
            const COMBINE_BLOCK_MAX: u32 = 256;

            const COMBINE_BLOCK_MIN: u32 = 32;

            if head_dim < COMBINE_BLOCK_MIN {
                COMBINE_BLOCK_MIN
            } else if head_dim > COMBINE_BLOCK_MAX {
                COMBINE_BLOCK_MAX
            } else {
                head_dim
            }
        }

        Launch::grid([rows, heads, 1], [combine_block(head_dim), 1, 1])
    }

    ctx.fire(
        Fire::at(
            "attn/dsv4_compress.cuh",
            crate::jit::symbol(&format!("::pie::attn::combine_attn_outputs<{}>", T::CPP)),
        )
        .apply(combine_attn(
            o_out.rows.unsigned_abs(),
            num_heads.unsigned_abs(),
            head_dim.unsigned_abs(),
        )),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o_out.arg(),
            lse_out.arg(),
            num_heads.arg(),
            head_dim.arg(),
        ],
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rows {
    pub start: i32,
    pub count: i32,
    pub total: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvScheme {
    Native = 0,
    Fp8PerTensor = 1,
    Int8PerTokenHead = 2,
    Fp8PerTokenHead = 3,
    Fp4Block = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum KvDType {
    Bf16 = 0,
    Fp16 = 1,
    Int8 = 3,
    Fp8E4M3 = 7,
    Fp8E5M2 = 8,
}

#[derive(Clone, Copy, Debug)]
pub struct KvLayer {
    pub k_pages: *mut c_void,
    pub v_pages: *mut c_void,
    pub page_size: i32,
    pub head_dim: i32,
    pub num_kv_heads: i32,
    pub hnd: bool,
    pub scheme: KvScheme,
    pub storage_dtype: KvDType,
    pub block_size: i32,
    pub num_pages: i32,
    pub k_scales: *mut c_void,
    pub v_scales: *mut c_void,
    pub k_bf16_pages: *mut c_void,
    pub v_bf16_pages: *mut c_void,
    pub k_env_min: *mut u16,
    pub k_env_max: *mut u16,
    pub has_envelopes: bool,
    pub is_native_bf16: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MlaLayer {
    pub ckv_pages: *mut c_void,
    pub kpe_pages: *mut c_void,
    pub page_size: i32,
    pub kv_lora_rank: i32,
    pub qk_rope_head_dim: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct AttnWorkspace {
    pub float_buffer: *mut c_void,
    pub float_bytes: usize,
    pub int_buffer: *mut c_void,
    pub int_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MlaPlan {
    pub info: crate::attn::plan::info::MlaPlanInfo,
    pub int_arena: *mut c_void,
    pub float_arena: *mut c_void,
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub qo_indptr: *const u32,
    pub kv_page_indices: *const u32,
    pub kv_page_indptr: *const u32,
    pub kv_last_page_lens: *const u32,
    pub row_valid: *const u8,
    pub requests: i32,
}
