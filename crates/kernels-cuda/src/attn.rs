use crate::jit::Ctx;
use crate::jit::Launch;
use core::ffi::c_void;
use kernels::Bind;
use kernels::Fire;
use kernels::Refusal;

use crate::jit::abi::Tensor;
use crate::views::KvCache;
use kernels::plane::{Cache, Const, In, Out};
use kernels::raises::Struct;

#[allow(unused_imports)]
use crate::jit::abi::bf16;
use kernels::plane::InOut;

use kernels::points::{Dtype, Element, Fan, Mark, Point, Prim, Shape, Slot, Width};

use kernels::points::Plane;

pub mod fa2;

pub mod plan;

fn width_of(n: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(n).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(n),
        max: i64::from(i32::MAX),
    })
}

const NO_SOFT_CAP: f32 = 0.0;

fn window_left(window: u32) -> Result<i32, Refusal> {
    if window == 0 {
        return Ok(-1);
    }
    width_of(window - 1, "the sliding window this statement states")
}

fn agrees(planned: i32, stated: u32) -> Result<(), Refusal> {
    if planned == width_of(stated, "a stated head width")? {
        return Ok(());
    }
    Err(Refusal::Narrow {
        what: "the head width this statement states is not the one this fire's \
               attention schedule was planned at",
        at: i64::from(stated),
    })
}

fn variant_agrees(planned_full: bool, window: u32) -> Result<(), Refusal> {
    if planned_full == (window == 0) {
        return Ok(());
    }
    Err(Refusal::Narrow {
        what: "the window this statement states is not the reading this fire's \
               attention schedule was planned for",
        at: i64::from(window),
    })
}

fn as_in<T: kernels::points::Scalar>(r: &In<Tensor<T>>) -> In<Tensor<bf16>> {
    In {
        ptr: r.ptr.cast::<bf16>(),
        rows: r.rows,
        width: r.width,
    }
}

fn as_out<T: kernels::points::Scalar>(r: &Out<Tensor<T>>) -> Out<Tensor<bf16>> {
    Out {
        ptr: r.ptr.cast::<bf16>(),
        rows: r.rows,
        width: r.width,
    }
}

fn row_valid_at(view: &crate::views::PagedKvView, rows: i32) -> In<Tensor<i32>> {
    In {
        ptr: view.row_valid.cast::<i32>(),
        rows,
        width: 1,
    }
}

#[allow(clippy::too_many_arguments)]
fn fa2_decode(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    pages: Cache<Struct<KvCache>>,
    window: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Out<Tensor<bf16>>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    let plan = ctx.raised_at::<crate::raises::Fa2Decode>(kernels::raises::Class::attention(
        head_dim, window,
    ))?;

    let planned = unsafe { &*plan.ptr };
    agrees(planned.head_dim, head_dim)?;
    variant_agrees(planned.full_attention_variant, window)?;
    fa2::dispatch_attention_flashinfer_decode(
        ctx,
        q,
        plan,
        o,
        Const::new(window_left(window)?),
        Const::new(NO_SOFT_CAP),
        Const::new(sm_scale),
        pages.raised(),
        lse,
    )
}

#[allow(clippy::too_many_arguments)]
fn fa2_prefill(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    indptr: In<Tensor<i32>>,
    pages: Cache<Struct<KvCache>>,
    window: u32,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Out<Tensor<bf16>>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<(), Refusal> {
    fa2::attention_flashinfer_prefill(
        ctx,
        q,
        o,
        Const::new(NO_SOFT_CAP),
        Const::new(sm_scale),
        pages.raised(),
        indptr,
        Const::new(width_of(head_dim, "the head width this attention states")?),
        ctx.raised::<crate::raises::Fa2Prefill>()?,
        ctx.raised::<crate::views::QoIndptrHost>()?,
        ctx.raised::<crate::views::KvPageIndptrHost>()?,
        Const::new(width_of(
            kv_heads,
            "the kv head count this attention states",
        )?),
        Const::new(window_left(window)?),
        lse,
    )
}

#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    fn decode<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.decode at an element other than bf16")?;
        fa2_decode(
            self,
            as_in(&q),
            pages,
            window,
            head_dim,
            sm_scale,
            as_out(&o),
            None,
        )
    }

    fn decode_lse<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
        lse: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.decode_lse at an element other than bf16")?;
        fa2_decode(
            self,
            as_in(&q),
            pages,
            window,
            head_dim,
            sm_scale,
            as_out(&o),
            Some(lse),
        )
    }

    fn prefill<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.prefill at an element other than bf16")?;
        fa2_prefill(
            self,
            as_in(&q),
            indptr,
            pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            as_out(&o),
            None,
        )
    }

    fn prefill_lse<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
        lse: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.prefill_lse at an element other than bf16")?;
        fa2_prefill(
            self,
            as_in(&q),
            indptr,
            pages,
            window,
            head_dim,
            kv_heads,
            sm_scale,
            as_out(&o),
            Some(lse),
        )
    }

    fn masked<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.masked at an element other than bf16")?;
        let plan = self.raised_at::<crate::raises::Fa2Prefill>(
            kernels::raises::Class::attention(head_dim, window),
        )?;

        let planned = unsafe { &*plan.ptr };
        agrees(planned.head_dim, head_dim)?;
        if planned.window_left >= 0 {
            return Err(Refusal::Narrow {
                what: "this fire's masked prefill schedule was carved for a windowed \
                       reading; the window rides the launch, so the schedule has to \
                       cover the whole prefix",
                at: i64::from(planned.window_left),
            });
        }
        fa2::dispatch_attention_flashinfer_prefill_custom(
            self,
            as_in(&q),
            plan,
            as_out(&o),
            Const::new(window_left(window)?),
            Const::new(NO_SOFT_CAP),
            Const::new(sm_scale),
            self.raised::<crate::views::AttnMask>()?,
            pages.raised(),
            indptr,
            None,
        )
    }

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        k: In<Tensor<T>>,
        v: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.kv_append at an element other than bf16")?;
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let (kv_heads, head_dim) = head_split(view, k.width)?;
        let (k, v) = (as_in(&k), as_in(&v));
        let csr = In {
            ptr: view.qo_indptr,
            rows: view.requests,
            width: 1,
        };

        let origin = In {
            ptr: core::ptr::null::<i32>(),
            rows: 1,
            width: 1,
        };
        if view.native_bf16 {
            kv_paged::write_kv_to_pages_bf16(
                self,
                k,
                v,
                row,
                Const::new(kv_heads),
                Const::new(head_dim),
                origin,
                csr,
                row_valid_at(view, k.rows),
            )
        } else {
            kv_paged::write_kv_to_pages_quantised(
                self,
                k,
                v,
                row,
                Const::new(kv_heads),
                Const::new(head_dim),
                origin,
                csr,
            )
        }
    }

    fn sink<T: kernels::points::Scalar>(
        &self,
        o: InOut<Tensor<T>>,
        lse: In<Tensor<f32>>,
        sink: Const<Tensor<T>>,
        head_dim: u32,
    ) -> Result<(), Refusal> {
        let head_dim = width_of(head_dim, "the head width this sink states")?;
        let dst = o.all("the row whose heads are counted")?;
        let heads = crate::norm::heads(&dst, head_dim)?;
        self.fire(
            Fire::at(
                "attn/attn_sink.cuh",
                crate::jit::symbol(&format!("::pie::attn::attn_sink_rescale<{}>", T::CPP)),
            )
            .apply(per_head_elementwise(
                o.rows.unsigned_abs(),
                heads.unsigned_abs(),
                head_dim.unsigned_abs(),
            )),
            &[
                o.arg(),
                lse.arg(),
                sink.arg(),
                o.rows.arg(),
                heads.arg(),
                head_dim.arg(),
            ],
        )
    }

    fn merge_lse<T: kernels::points::Scalar>(
        &self,
        o1: In<Tensor<T>>,
        lse1: In<Tensor<f32>>,
        o2: In<Tensor<T>>,
        lse2: In<Tensor<f32>>,
        heads: u32,
        head_dim: u32,
        o: Out<Tensor<T>>,
        lse: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let heads = width_of(heads, "the head count this merge states")?;
        let head_dim = width_of(head_dim, "the head width this merge states")?;

        const COMBINE_BLOCK_MIN: u32 = 32;
        const COMBINE_BLOCK_MAX: u32 = 256;
        self.fire(
            Fire::at(
                "attn/dsv4_compress.cuh",
                crate::jit::symbol(&format!("::pie::attn::combine_attn_outputs<{}>", T::CPP)),
            )
            .apply(Launch::grid(
                [o.rows.unsigned_abs(), heads.unsigned_abs(), 1],
                [
                    head_dim
                        .unsigned_abs()
                        .clamp(COMBINE_BLOCK_MIN, COMBINE_BLOCK_MAX),
                    1,
                    1,
                ],
            )),
            &[
                o1.arg(),
                lse1.arg(),
                o2.arg(),
                lse2.arg(),
                o.arg(),
                lse.arg(),
                heads.arg(),
                head_dim.arg(),
            ],
        )
    }

    fn logit_softcap<T: kernels::points::Scalar>(
        &self,
        x: InOut<Tensor<T>>,
        cap: f32,
    ) -> Result<(), Refusal> {
        let n = softcap_elems(&x.all("out_width(0)")?)?;
        self.fire(
            Fire::at(
                "attn/softcap.cuh",
                crate::jit::symbol(&format!("::pie::attn::logit_softcap<{}>", T::CPP)),
            )
            .apply(softcap_launch(cap, n)?),
            &[x.arg(), cap.arg(), n.arg()],
        )
    }

    fn kv_append_shared<T: kernels::points::Scalar>(
        &self,
        plane: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("attention.kv_append_shared at an element other than bf16")?;
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let (kv_heads, head_dim) = head_split(view, plane.width)?;

        let shared = In {
            ptr: plane.ptr.cast::<bf16>(),
            rows: plane.rows,
            width: plane.width,
        };
        let csr = In {
            ptr: view.qo_indptr,
            rows: view.requests,
            width: 1,
        };

        let origin = In {
            ptr: core::ptr::null::<i32>(),
            rows: 1,
            width: 1,
        };
        if view.native_bf16 {
            kv_paged::write_kv_to_pages_bf16(
                self,
                shared,
                shared,
                row,
                Const::new(kv_heads),
                Const::new(head_dim),
                origin,
                csr,
                In {
                    ptr: view.row_valid.cast::<i32>(),
                    rows: plane.rows,
                    width: 1,
                },
            )
        } else {
            kv_paged::write_kv_to_pages_quantised(
                self,
                shared,
                shared,
                row,
                Const::new(kv_heads),
                Const::new(head_dim),
                origin,
                csr,
            )
        }
    }
}

fn head_split(view: &crate::views::PagedKvView, row: i32) -> Result<(i32, i32), Refusal> {
    const WHAT: &str = "the head width this pool row's strides spell";
    let wide = if view.layout != 0 {
        view.seq_stride
    } else {
        view.head_stride
    };
    let head_dim = i32::try_from(wide).map_err(|_| Refusal::Wide {
        what: WHAT,
        at: wide,
        max: i64::from(i32::MAX),
    })?;
    if head_dim <= 0 {
        return Err(Refusal::Empty { what: WHAT });
    }
    if row <= 0 || row % head_dim != 0 {
        return Err(Refusal::Narrow {
            what: "the appended plane does not divide by the pool row's head width",
            at: i64::from(row),
        });
    }
    Ok((row / head_dim, head_dim))
}

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

pub mod attention_flashinfer {
    use crate::jit::{Ctx, Launch};
    use kernels::plane::Fire;

    use crate::jit::abi::Tensor;
    use kernels::Refusal;
    use kernels::plane::{In, Out};

    use kernels::Bind;

    #[allow(clippy::too_many_arguments)]
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

pub mod attention_naive {
    pub const BLOCK: u32 = 256;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_mla_to_pages(
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

        pub top_k: i32,
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

        pub selection: *const i32,
    }

    #[must_use]
    pub enum NaivePlan {
        Scalar { launch: Launch, head_group: i32 },
        Mma { launch: Launch },
        Declined(NaiveDecline),
    }

    pub fn plan(shape: NaiveShape, have_indptr: bool, selected: bool) -> NaivePlan {
        pub const MMA_THREADS: u32 = 256;

        if shape.total_tokens <= 0 {
            return NaivePlan::Declined(NaiveDecline::NoTokens);
        }

        if !have_indptr {
            return NaivePlan::Declined(NaiveDecline::MissingIndptr);
        }

        if !selected && mma_supported(shape.kv_lora_rank, shape.qk_rope_head_dim, shape.num_heads) {
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
        match plan(shape, have_indptr, !ptrs.selection.is_null()) {
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
                        ptrs.selection.arg(),
                        shape.top_k.arg(),
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

    #[must_use]
    pub unsafe fn pack(
        plan: &MlaPlanInfo,
        shape: Shape,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
            unsafe { base.offset(offset as isize).cast::<T>() }
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

#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_mla_bf16(
    ctx: &Ctx<'_>,
    plan: &MlaPlan,
    q_nope: In<Tensor<bf16>>,
    q_pe: In<Tensor<bf16>>,
    layer: MlaLayer,
    o: Out<Tensor<bf16>>,
    num_heads: i32,
    sm_scale: Const<f32>,
    causal: bool,
    kvc: In<Struct<KvCache>>,
    qo_indptr: *const u32,
    num_requests: Const<i32>,
    lse: Option<Out<Tensor<f32>>>,
) -> Result<MlaDispatch, Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    let kvc = unsafe { &*kvc.ptr };
    let kv_page_indices = kvc.page_indices as *const u32;
    let kv_page_indptr = kvc.page_indptr as *const u32;

    let lse = lse.map_or(core::ptr::null_mut(), |l| l.ptr);
    let kv_last_page_lens = kvc.last_page_lens as *const u32;
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
            selection: core::ptr::null(),
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
            top_k: 0,
        };
        return mla_naive::fire(ctx, ptrs, shape).map(MlaDispatch::Naive);
    }

    let Some(arm) = mla_fa2::arm_index(fa2::plan::fa_device().max_smem_per_sm) else {
        return Err(Refusal::Absent {
            what: "a `DISPATCH_SMEM_CONFIG` arm for this device's shared memory per SM",
        });
    };

    if mla_fa2::ARMS[arm].cta_tile_kv < 32 {
        return Err(Refusal::Absent {
            what: "a latent attention arm this device can run: the only \
                   `DISPATCH_SMEM_CONFIG` arm that fits its shared memory is \
                   `CTA_TILE_KV = 16`, which writes past its own \
                   `SharedStorage` (measured; see this call site)",
        });
    }
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

fn mla_layer(kvc: &crate::views::PagedKvView, kv_lora_rank: i32, rope_dim: i32) -> MlaLayer {
    MlaLayer {
        ckv_pages: kvc.keys.cast::<c_void>(),
        kpe_pages: kvc.values.cast::<c_void>(),
        page_size: kvc.page_size,
        kv_lora_rank,
        qk_rope_head_dim: rope_dim,
    }
}

fn rope_per_head(q_pe: &In<Tensor<bf16>>, heads: i32) -> Result<i32, Refusal> {
    if heads <= 0 {
        return Err(Refusal::Empty {
            what: "the head count this attention states",
        });
    }
    if q_pe.width <= 0 || q_pe.width % heads != 0 {
        return Err(Refusal::Narrow {
            what: "the rotated half does not divide by the stated head count",
            at: i64::from(q_pe.width),
        });
    }
    Ok(q_pe.width / heads)
}

fn mla_plan_of(plan: &In<Struct<crate::raises::MlaPlanned>>) -> Result<&MlaPlan, Refusal> {
    if plan.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the latent attention plan this statement names",
        });
    }
    Ok(unsafe { &*plan.ptr })
}

fn kv_view_of(kvc: &In<Struct<KvCache>>) -> Result<&crate::views::PagedKvView, Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    Ok(unsafe { &*kvc.ptr })
}

#[allow(clippy::too_many_arguments)]
pub fn attention_mla_decode_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    plan: In<Struct<crate::raises::MlaPlanned>>,
    q_pe: In<Tensor<bf16>>,
    o: Out<Tensor<bf16>>,
    kvc: In<Struct<KvCache>>,
    heads: Const<i32>,
    kv_lora_rank: Const<i32>,
    sm_scale: Const<f32>,
) -> Result<(), Refusal> {
    let view = kv_view_of(&kvc)?;
    let heads = *heads;
    let rope = rope_per_head(&q_pe, heads)?;
    let layer = mla_layer(view, *kv_lora_rank, rope);
    dispatch_attention_mla_bf16(
        ctx,
        mla_plan_of(&plan)?,
        q,
        q_pe,
        layer,
        o,
        heads,
        sm_scale,
        false,
        kvc,
        view.qo_indptr as *const u32,
        Const::new(view.requests),
        None,
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
fn selected_attention_mla_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    q_pe: In<Tensor<bf16>>,
    selection: In<Tensor<i32>>,
    layer: MlaLayer,
    o: Out<Tensor<bf16>>,
    view: &crate::views::PagedKvView,
    qo_indptr: *const i32,
    num_requests: i32,
    num_heads: i32,
    sm_scale: f32,
    causal: bool,
) -> Result<(), Refusal> {
    if selection.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the selection this attention attends over",
        });
    }

    let sel = selection.all("the selection this attention attends over")?;
    let top_k = sel.width;

    if sel.rows != o.rows {
        return Err(Refusal::Narrow {
            what: "the selection does not carry one row per query row",
            at: i64::from(sel.rows),
        });
    }
    let ptrs = mla_naive::NaivePtrs {
        q_nope: q.ptr,
        q_pe: q_pe.ptr,
        ckv_pages: layer.ckv_pages.cast::<bf16>().cast_const(),
        kpe_pages: layer.kpe_pages.cast::<bf16>().cast_const(),
        qo_indptr: qo_indptr as *const u32,
        kv_page_indices: view.page_indices as *const u32,
        kv_page_indptr: view.page_indptr as *const u32,
        kv_last_page_lens: view.last_page_lens as *const u32,
        o: o.ptr,
        selection: selection.ptr,
    };
    let shape = mla_naive::NaiveShape {
        kv_lora_rank: layer.kv_lora_rank,
        qk_rope_head_dim: layer.qk_rope_head_dim,
        page_size: layer.page_size,
        total_tokens: o.rows,
        num_requests,
        num_heads,
        sm_scale,
        causal,
        top_k,
    };
    match mla_naive::fire(ctx, ptrs, shape)? {
        mla_naive::MlaNaive::Declined(why) => Err(match why {
            mla_naive::NaiveDecline::NoTokens => Refusal::Empty {
                what: "the query this selected attention was handed",
            },
            mla_naive::NaiveDecline::MissingIndptr => Refusal::Null {
                what: "the CSR triple this selected attention resolves its pages from",
            },
            mla_naive::NaiveDecline::UnsupportedKvLoraRank => Refusal::Narrow {
                what: "the latent rank this kernel can lane-split (a multiple of 32, at most 512)",
                at: i64::from(shape.kv_lora_rank),
            },
            mla_naive::NaiveDecline::UnsupportedRopeDim => Refusal::Narrow {
                what: "the rope width this kernel can lane-split (a multiple of 32, at most 128)",
                at: i64::from(shape.qk_rope_head_dim),
            },
        }),

        mla_naive::MlaNaive::LaunchedScalar | mla_naive::MlaNaive::LaunchedMma => Ok(()),
    }
}

pub mod qkv_fused {}

#[kernels_macros::claims]
impl Ctx<'_> {
    #[shape(q = [packed.rows, packed.width - 2 * kv_heads * head_dim])]
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_fused_qknorm_rope_vnorm_write<T: kernels::points::Scalar>(
        &self,
        packed: In<<Self as Plane>::Tensor<T>>,
        positions: In<<Self as Plane>::Tensor<i32>>,
        q_weight: Const<<Self as Plane>::Tensor<T>>,
        q_eps: f32,
        k_weight: Const<<Self as Plane>::Tensor<T>>,
        k_eps: f32,
        pages: Cache<<Self as Plane>::Pages>,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        q: Out<<Self as Plane>::Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("cuda::qkv_fused_qknorm_rope_vnorm_write at an element other than bf16")?;
        if q_eps != k_eps {
            return Err(Refusal::Unstated {
                what: "two head-norm epsilons on a fused write: the kernel applies one to \
                       both norms, so serving this would normalise k at q's epsilon",
            });
        }
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        let head_dim = width_of(head_dim, "the head width this fused write states")?;
        let kv_heads = width_of(kv_heads, "the kv head count this fused write states")?;
        if head_dim <= 0 {
            return Err(Refusal::Empty {
                what: "the head width this fused write states",
            });
        }
        let (packed, q_out) = (as_in(&packed), as_out(&q));

        let width = packed.all("the packed qkv row this write splits")?.width;
        let num_q_heads = (width - 2 * kv_heads * head_dim) / head_dim;
        if num_q_heads <= 0 {
            return Err(Refusal::Narrow {
                what: "a packed qkv row with no q plane left after its two kv planes",
                at: i64::from(width),
            });
        }
        let heads = num_q_heads.unsigned_abs() + kv_heads.unsigned_abs();
        let rows = packed.rows;

        let k_pages = view.keys.cast::<bf16>();
        let v_pages = view.values.cast::<bf16>();
        let page_indices = view.page_indices as *const u32;
        let page_indptr = view.page_indptr as *const u32;
        let last_page_lens = view.last_page_lens as *const u32;
        let w_page = view.write_page as *const u32;
        let w_off = view.write_offset as *const u32;
        let row_valid = view.row_valid;

        let rope_table = core::ptr::null::<f32>();
        let win = core::ptr::null::<u32>();
        let page_size = view.page_size;
        let hnd_layout = view.layout != 0;

        const WARP_BLOCK: u32 = 256;
        const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;
        const DECODE_BLOCK: u32 = 128;
        let warped = match head_dim {
            64 => Some(
                "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(64), false>",
            ),
            128 => Some(
                "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(128), false>",
            ),
            256 => Some(
                "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv_warp<::pie::i32(256), false>",
            ),
            _ => None,
        };
        if let Some(instantiation) = warped {
            let units = rows.unsigned_abs().saturating_mul(heads);
            return self.fire(
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
                    page_indices.arg(),
                    page_indptr.arg(),
                    last_page_lens.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    row_valid.arg(),
                    win.arg(),
                    rows.arg(),
                    num_q_heads.arg(),
                    kv_heads.arg(),
                    page_size.arg(),
                    hnd_layout.arg(),
                    theta.arg(),
                    q_eps.arg(),
                ],
            );
        }
        self.fire(
            Fire::at(
                "attn/qkv_fused.cuh",
                "::pie::attn::qkv_decode_qk_norm_rope_vnorm_write_kv<::pie::i32(128), false>",
            )
            .apply(Launch::grid(
                [rows.unsigned_abs(), heads, 1],
                [DECODE_BLOCK, 1, 1],
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
                page_indices.arg(),
                page_indptr.arg(),
                last_page_lens.arg(),
                w_page.arg(),
                w_off.arg(),
                row_valid.arg(),
                win.arg(),
                num_q_heads.arg(),
                kv_heads.arg(),
                head_dim.arg(),
                page_size.arg(),
                hnd_layout.arg(),
                theta.arg(),
                q_eps.arg(),
            ],
        )
    }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "both are guarded positive by every caller"
)]
fn route_rows(rows: i32, width: i32) -> Launch {
    let (rows, width) = (rows as u32, width as u32);
    Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
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

    use kernels::plane::{Const, In, Out};
    use kernels::raises::Struct;
    use kernels::{Bind, Fire};

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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_kv_to_pages_bf16(
        ctx: &Ctx<'_>,
        k_curr: In<Tensor<bf16>>,
        v_curr: In<Tensor<bf16>>,
        kvc: In<Struct<KvCache>>,
        num_kv_heads: Const<i32>,

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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_kv_to_pages_quantised(
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dequant_kv_cache_layer_to_bf16_active(
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

#[allow(clippy::too_many_arguments)]
pub fn attn_res_blend<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    prefix: In<Tensor<T>>,
    blocks: In<Tensor<T>>,

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

pub(crate) fn kimi_split_kv_a_norm<T: crate::RoutineElem>(
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

fn width(what: &'static str, v: u32) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if T::CPP == <bf16 as kernels::Elem>::CPP {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

#[kernels_macros::claims]
impl kernels::points::Mla for Ctx<'_> {
    fn latents<T: kernels::points::Scalar>(
        &self,
        kv_a: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Out<Tensor<T>>,
        k_pe: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = kv_lora_rank;
        kimi_split_kv_a_norm(self, kv_a, weight, kv_c, k_pe, Const::new(eps))
    }

    fn latents_rope<T: kernels::points::Scalar>(
        &self,
        kv_a: In<Tensor<T>>,
        positions: In<Tensor<i32>>,
        weight: Const<Tensor<T>>,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: Out<Tensor<T>>,
        k_pe: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.latents_rope at an element other than bf16")?;

        let _ = kv_lora_rank;
        let rope = width("the rope width this cut states", rope_dim)?;

        let rotated = InOut {
            ptr: k_pe.ptr.cast::<bf16>(),
            rows: k_pe.rows,
            width: k_pe.width,
        };
        kimi_split_kv_a_norm(self, kv_a, weight, kv_c, k_pe, Const::new(eps))?;

        crate::rope::rope_partial_q_bf16(self, rotated, rope, rope, theta, positions.ptr)
    }

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        kv_c: In<Tensor<T>>,
        k_pe: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.kv_append at an element other than bf16")?;
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let kv_lora_rank = kv_c.all("the latent row this append writes")?.width;
        let rope = k_pe.all("the rotated row this append writes")?.width;
        write_mla_to_pages(
            self,
            mla_layer(view, kv_lora_rank, rope),
            In {
                ptr: kv_c.ptr.cast::<bf16>(),
                rows: kv_c.rows,
                width: kv_c.width,
            },
            In {
                ptr: k_pe.ptr.cast::<bf16>(),
                rows: k_pe.rows,
                width: k_pe.width,
            },
            In {
                ptr: view.qo_indptr,
                rows: view.requests,
                width: 1,
            },
            pages.raised(),
            In {
                ptr: view.row_valid.cast::<i32>(),
                rows: kv_c.rows,
                width: 1,
            },
            Const::new(view.requests),
        )
    }

    fn attention_decode<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        q_pe: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.attention_decode at an element other than bf16")?;
        attention_mla_decode_bf16(
            self,
            as_in(&q),
            self.raised::<crate::raises::MlaPlanned>()?,
            as_in(&q_pe),
            as_out(&o),
            pages.raised(),
            Const::new(width("the head count this attention states", heads)?),
            Const::new(width(
                "the latent rank this attention states",
                kv_lora_rank,
            )?),
            Const::new(sm_scale),
        )
    }

    fn attention_prefill<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        q_pe: In<Tensor<T>>,
        pages: Cache<Struct<KvCache>>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.attention_prefill at an element other than bf16")?;
        let q = as_in(&q);
        let plan = self.raised::<crate::raises::MlaPlanned>()?;
        let q_pe = as_in(&q_pe);
        let o = as_out(&o);
        let kvc = pages.raised();
        let heads = width("the head count this attention states", heads)?;
        let kv_lora_rank = width("the latent rank this attention states", kv_lora_rank)?;
        let view = kv_view_of(&kvc)?;
        let rope = rope_per_head(&q_pe, heads)?;
        let layer = mla_layer(view, kv_lora_rank, rope);

        let num_requests = indptr.rows;
        dispatch_attention_mla_bf16(
            self,
            mla_plan_of(&plan)?,
            q,
            q_pe,
            layer,
            o,
            heads,
            Const::new(sm_scale),
            true,
            kvc,
            indptr.ptr as *const u32,
            Const::new(num_requests),
            None,
        )
        .map(|_| ())
    }

    fn split_q_b<T: kernels::points::Scalar>(
        &self,
        q_b: In<Tensor<T>>,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Out<Tensor<T>>,
        q_pe: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let heads = width("the head count this cut states", heads)?;
        let nope = width("the nope width this cut states", nope_dim)?;
        let rope = width("the rope width this cut states", rope_dim)?;
        let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
        let total = i64::from(q_b.rows) * width;
        if total > i64::from(i32::MAX) {
            return Err(Refusal::Wide {
                what: "rows",
                at: i64::from(q_b.rows),
                max: i64::from(i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX)),
            });
        }
        let total = total as i32;

        self.fire(
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

    fn absorb_q<T: kernels::points::Scalar>(
        &self,
        q_nope: In<Tensor<T>>,
        kv_b: Const<Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.absorb_q at an element other than bf16")?;
        crate::gemm::absorb::mla_absorb_q_to_latent_bf16(
            self,
            In {
                ptr: q_nope.ptr.cast::<c_void>(),
                rows: q_nope.rows,
                width: q_nope.width,
            },
            Const::new(kv_b.v.cast::<c_void>()),
            Out {
                ptr: q_latent.ptr.cast::<c_void>(),
                rows: q_latent.rows,
                width: q_latent.width,
            },
            Const::new(width("the head count this absorb states", heads)?),
            Const::new(width("the nope width this absorb states", nope_dim)?),
            Const::new(width("the value width this absorb states", v_head_dim)?),
            Const::new(width("the latent rank this absorb states", kv_lora_rank)?),
            Const::new(q_nope.rows),
        )
    }

    fn absorb_out<T: kernels::points::Scalar>(
        &self,
        latent: In<Tensor<T>>,
        kv_b: Const<Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.absorb_out at an element other than bf16")?;
        crate::gemm::absorb::mla_absorb_latent_to_v_bf16(
            self,
            In {
                ptr: latent.ptr.cast::<c_void>(),
                rows: latent.rows,
                width: latent.width,
            },
            Const::new(kv_b.v.cast::<c_void>()),
            Out {
                ptr: o.ptr.cast::<c_void>(),
                rows: o.rows,
                width: o.width,
            },
            Const::new(width("the head count this absorb states", heads)?),
            Const::new(width("the nope width this absorb states", nope_dim)?),
            Const::new(width("the value width this absorb states", v_head_dim)?),
            Const::new(width("the latent rank this absorb states", kv_lora_rank)?),
            Const::new(latent.rows),
        )
    }

    fn attention_decode_selected<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        q_pe: In<Tensor<T>>,
        selection: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.attention_decode_selected at an element other than bf16")?;
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let heads = width("the head count this attention states", heads)?;
        let q_pe = In {
            ptr: q_pe.ptr.cast::<bf16>(),
            rows: q_pe.rows,
            width: q_pe.width,
        };
        let rope = rope_per_head(&q_pe, heads)?;
        selected_attention_mla_bf16(
            self,
            In {
                ptr: q.ptr.cast::<bf16>(),
                rows: q.rows,
                width: q.width,
            },
            q_pe,
            selection,
            mla_layer(
                view,
                width("the latent rank this attention states", kv_lora_rank)?,
                rope,
            ),
            Out {
                ptr: o.ptr.cast::<bf16>(),
                rows: o.rows,
                width: o.width,
            },
            view,
            view.qo_indptr,
            view.requests,
            heads,
            sm_scale,
            false,
        )
    }

    fn attention_prefill_selected<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        q_pe: In<Tensor<T>>,
        selection: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("mla.attention_prefill_selected at an element other than bf16")?;
        let row = pages.raised();
        let view = kv_view_of(&row)?;
        let heads = width("the head count this attention states", heads)?;
        let q_pe = In {
            ptr: q_pe.ptr.cast::<bf16>(),
            rows: q_pe.rows,
            width: q_pe.width,
        };
        let rope = rope_per_head(&q_pe, heads)?;
        selected_attention_mla_bf16(
            self,
            In {
                ptr: q.ptr.cast::<bf16>(),
                rows: q.rows,
                width: q.width,
            },
            q_pe,
            selection,
            mla_layer(
                view,
                width("the latent rank this attention states", kv_lora_rank)?,
                rope,
            ),
            Out {
                ptr: o.ptr.cast::<bf16>(),
                rows: o.rows,
                width: o.width,
            },
            view,
            indptr.ptr,
            indptr.rows,
            heads,
            sm_scale,
            true,
        )
    }
}

#[kernels_macros::claims]
impl kernels::points::Index for Ctx<'_> {
    fn layernorm_rope<T: kernels::points::Scalar>(
        &self,
        k: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        weight: Const<Tensor<T>>,
        bias: Const<Tensor<T>>,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("index.layernorm_rope at an element other than bf16")?;
        let idx_k: InOut<Tensor<bf16>> = InOut {
            ptr: k.ptr.cast::<bf16>(),
            rows: k.rows,
            width: k.width,
        };
        let k_norm_weight: Const<Tensor<bf16>> = Const::new(weight.v.cast::<bf16>());
        let k_norm_bias: Const<Tensor<bf16>> = Const::new(bias.v.cast::<bf16>());
        let rope_dim = width("the rope width this norm states", rope_dim)?;

        let positions = positions.ptr;

        let dst = idx_k.all("out_width(0)")?;
        let head_dim = dst.width;

        self.fire(
            Fire::at(
                "attn/dsa_indexer.cuh",
                "::pie::attn::index_knorm_rope<::pie::bf16>",
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

    fn rope<T: kernels::points::Scalar>(
        &self,
        q: InOut<Tensor<T>>,
        positions: In<Tensor<i32>>,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("index.rope at an element other than bf16")?;
        let idx_q: InOut<Tensor<bf16>> = InOut {
            ptr: q.ptr.cast::<bf16>(),
            rows: q.rows,
            width: q.width,
        };
        let n_heads = width("the head count this rotation states", heads)?;
        let head_dim = width("the head width this rotation states", head_dim)?;
        let rope_dim = width("the rope width this rotation states", rope_dim)?;

        let positions = positions.ptr;

        self.fire(
            Fire::at(
                "attn/dsa_indexer.cuh",
                "::pie::attn::index_q_rope<::pie::bf16>",
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

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        k: In<Tensor<T>>,
        keys: Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("index.kv_append at an element other than bf16")?;
        let row = keys.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let dst = k.all("the index key row this append writes")?;
        index_pool_pitch(view, dst.width)?;
        write_mla_to_pages(
            self,
            MlaLayer {
                ckv_pages: view.keys.cast::<c_void>(),

                kpe_pages: core::ptr::null_mut(),
                page_size: view.page_size,
                kv_lora_rank: dst.width,
                qk_rope_head_dim: 0,
            },
            In {
                ptr: k.ptr.cast::<bf16>(),
                rows: k.rows,
                width: dst.width,
            },
            In {
                ptr: core::ptr::null::<bf16>(),
                rows: k.rows,
                width: 0,
            },
            In {
                ptr: view.qo_indptr,
                rows: view.requests,
                width: 1,
            },
            row,
            In {
                ptr: view.row_valid.cast::<i32>(),
                rows: k.rows,
                width: 1,
            },
            Const::new(view.requests),
        )
    }

    fn topk<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        weights: In<Tensor<T>>,
        keys: Cache<Struct<KvCache>>,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        selection: Out<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("index.topk at an element other than bf16")?;
        let row = keys.raised();
        let view = kv_view_of(&row)?;
        if view.qo_indptr.is_null() {
            return Err(Refusal::Null {
                what: "the query CSR this fire's pool row carries",
            });
        }
        let heads = width("the head count this ranking states", heads)?;
        let head_dim = width("the key width this ranking states", head_dim)?;
        let top_k = width("the selection budget this ranking states", top_k)?;

        index_pool_pitch(view, head_dim)?;
        let q = q.all("the index query this ranking scores")?;
        if q.width != heads.saturating_mul(head_dim) {
            return Err(Refusal::Narrow {
                what: "the index query does not divide by the stated head count and width",
                at: i64::from(q.width),
            });
        }
        let w = weights.all("the index head weights this ranking scores with")?;
        if w.width != heads {
            return Err(Refusal::Narrow {
                what: "the index head weights are not one per stated head",
                at: i64::from(w.width),
            });
        }
        let out = selection.all("the selection this ranking writes")?;
        if out.width != top_k {
            return Err(Refusal::Narrow {
                what: "the selection this statement allocated is not the budget it stated",
                at: i64::from(out.width),
            });
        }

        let max_kv = view
            .max_pages_per_request
            .checked_mul(view.page_size)
            .filter(|bound| *bound > 0)
            .ok_or(Refusal::Empty {
                what: "the page budget this fire's pool row states",
            })?;
        let scores = self
            .scratch(
                "attn::dsa_index_scores",
                (out.rows as usize)
                    .saturating_mul(max_kv as usize)
                    .saturating_mul(core::mem::size_of::<f32>()),
            )?
            .cast::<f32>();

        self.fire(
            Fire::at(
                "attn/dsa_indexer.cuh",
                "::pie::attn::index_topk_paged<::pie::bf16>",
            )
            .apply(Launch::per_row(
                out.rows.unsigned_abs(),
                dsa_indexer::K_BLOCK,
            )),
            &[
                q.ptr.cast::<bf16>().arg(),
                w.ptr.cast::<bf16>().arg(),
                view.keys.cast::<bf16>().cast_const().arg(),
                (view.qo_indptr as *const u32).arg(),
                (view.page_indices as *const u32).arg(),
                (view.page_indptr as *const u32).arg(),
                (view.last_page_lens as *const u32).arg(),
                scores.arg(),
                selection.ptr.arg(),
                view.requests.arg(),
                heads.arg(),
                head_dim.arg(),
                view.page_size.arg(),
                max_kv.arg(),
                top_k.arg(),
            ],
        )
    }
}

fn index_pool_pitch(view: &crate::views::PagedKvView, row: i32) -> Result<(), Refusal> {
    const WHAT: &str = "the token pitch this index pool's strides spell";
    if view.layout != 0 {
        return Err(Refusal::Absent {
            what: "a contiguous index-key append into an HND pool: a token step there is \
                   one head wide and the row would have to be scattered",
        });
    }
    if row <= 0 {
        return Err(Refusal::Empty { what: WHAT });
    }
    if view.seq_stride != i64::from(row) {
        return Err(Refusal::Narrow {
            what: WHAT,
            at: view.seq_stride,
        });
    }
    Ok(())
}

const fn compressor_coff(ratio: i32) -> i32 {
    if ratio == 4 { 2 } else { 1 }
}

fn boundary_rope(ctx: &Ctx<'_>, rows: i32) -> Result<Out<Tensor<i32>>, Refusal> {
    let bytes = usize::try_from(rows.max(0)).unwrap_or(0) * core::mem::size_of::<i32>();
    let ptr = ctx.scratch("attn::dsv4_boundary_rope", bytes)?;
    Ok(Out {
        ptr: ptr.cast::<i32>(),
        rows,
        width: 1,
    })
}

fn row_valid_staged(ctx: &Ctx<'_>, rows: i32) -> In<Tensor<i32>> {
    In {
        ptr: ctx.staged::<crate::views::RowValid>().ptr.cast::<i32>(),
        rows,
        width: 1,
    }
}

#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {
    fn boundary_decode(
        &self,
        positions: In<Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Tensor<i32>>,
        boundary_req: Out<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        let rows = boundary_pos.rows;
        let out_pos = boundary_pos;
        let out_req = boundary_req;
        let out_rope = boundary_rope(self, rows)?;
        let ratio = width("the pooling ratio this statement states", ratio)?;
        let row_valid = row_valid_staged(self, rows);
        let row_valid = row_valid.ptr as *const u8;
        if ratio <= 0 {
            return Err(Refusal::Narrow {
                what: "ratio",
                at: i64::from(ratio),
            });
        }

        self.fire(
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

    fn boundary_prefill(
        &self,
        positions: In<Tensor<i32>>,
        indptr: In<Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Tensor<i32>>,
        boundary_req: Out<Tensor<i32>>,
    ) -> Result<(), Refusal> {
        let rows = boundary_pos.rows;
        let out_pos = boundary_pos;
        let out_req = boundary_req;
        let out_rope = boundary_rope(self, rows)?;
        let ratio = width("the pooling ratio this statement states", ratio)?;
        let row_valid = row_valid_staged(self, rows);
        let qo_indptr = indptr;
        let row_valid = row_valid.ptr as *const u8;

        let num_requests = qo_indptr.rows;
        let qo_indptr = qo_indptr.ptr as *const u32;
        if ratio <= 0 {
            return Err(Refusal::Narrow {
                what: "ratio",
                at: i64::from(ratio),
            });
        }

        self.fire(
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

    fn gather<T: kernels::points::Scalar>(
        &self,
        boundary_pos: In<Tensor<i32>>,
        boundary_req: In<Tensor<i32>>,
        pages: Cache<Struct<KvCache>>,
        head_dim: u32,
        ratio: u32,
        entries: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("pool.gather at an element other than bf16")?;
        if entries.width != width("the head width this gather states", head_dim)? {
            return Err(Refusal::Narrow {
                what: "the head width this statement states is not the width of the \
                       entry it sized",
                at: i64::from(entries.width),
            });
        }
        let ratio = width("the pooling ratio this statement states", ratio)?;
        let out = as_out(&entries);
        let coff = compressor_coff(ratio);
        let kvc = pages.raised();
        let state_kv = self.raised::<crate::views::Dsv4StateKv>()?;
        let state_score = self.raised::<crate::views::Dsv4StateScore>()?;
        let ape = self.raised::<crate::views::Dsv4Ape>()?;
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let num_entries = boundary_pos.rows;
        let page_size = kvc.page_size;
        let kv_page_indices = kvc.page_indices as *const u32;
        let kv_page_indptr = kvc.page_indptr as *const u32;
        let state_kv = state_kv.ptr;
        let state_score = state_score.ptr;
        let ape = ape.ptr;
        let head_dim = out.all("out_width(0)")?.width;

        self.fire(
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

    fn kv_append<T: kernels::points::Scalar>(
        &self,
        entries: In<Tensor<T>>,
        boundary_pos: In<Tensor<i32>>,
        boundary_req: In<Tensor<i32>>,
        pool: Cache<Struct<KvCache>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("pool.kv_append at an element other than bf16")?;
        let entries = as_in(&entries);
        let kvc = pool.raised();
        let comp_kv = self.raised::<crate::views::Dsv4CompKvPages>()?;
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

        self.fire(
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

    fn attention_lse<T: kernels::points::Scalar>(
        &self,
        q: In<Tensor<T>>,
        positions: In<Tensor<i32>>,
        entries: Cache<Struct<KvCache>>,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Tensor<T>>,
        lse: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("pool.attention_lse at an element other than bf16")?;
        let request_of_token = self.raised::<crate::views::RequestOfToken>()?;
        let q = as_in(&q);
        let o = as_out(&o);
        let lse_out = lse;
        let ratio = width("the pooling ratio this statement states", ratio)?;
        let num_q_heads = width("the head count this attention states", heads)?;
        let head_dim = width("the head width this attention states", head_dim)?;
        let kvc = entries.raised();
        let request_of_token: In<Tensor<i32>> = In {
            ptr: request_of_token.ptr,
            rows: o.rows,
            width: 1,
        };
        let comp_kv = self.raised::<crate::views::Dsv4CompKvPages>()?;
        if kvc.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the kv view this statement names",
            });
        }
        let kvc = unsafe { &*kvc.ptr };
        let page_size = kvc.page_size;

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

        self.fire(
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
}
