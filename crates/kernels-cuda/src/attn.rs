use crate::jit::Ctx;
use crate::jit::Launch;
use core::ffi::c_void;
use kernels::Bind;
use kernels::Fire;
use kernels::Refusal;
use kernels_macros::routine;

use crate::jit::abi::Tensor;
use crate::views::{Dsv4CompKvPages, KvCache, MtpPendingHidden, RecurrentState};
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, Out};

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

/// A stated `u32` as the `i32` the routines take.
fn width_of(n: u32, what: &'static str) -> Result<i32, Refusal> {
    i32::try_from(n).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(n),
        max: i64::from(i32::MAX),
    })
}

/// The `Attention` family, claimed. Five of eleven points land, and the
/// six absences are the migration's most deliberate: THE FA2 CORE IS
/// CLAIM-ONLY BY DESIGN.
///
/// * `attention.decode` / `attention.decode_lse` — the routine takes
///   `plan: In<Struct<Fa2Decode>>`, the decode PLAN CACHE: a resident this
///   plane builds at load, uploads per fire, and reads the split-kv
///   partition out of. A statement carries the query, the page row and
///   three numbers; nothing declared can conjure a plan cache, and a body
///   that reached for one would be staging on the operand column's behalf.
///   The routines keep their own `canon` and the points resolve through it.
/// * `attention.prefill` / `attention.prefill_lse` — the same seam plus
///   two HOST MIRRORS. `attention_flashinfer_prefill` plans in-body and
///   wants `qo_indptr_host` and `kv_page_indptr_host` beside the device
///   CSR, because the partition arithmetic runs on the host before the
///   launch. A host mirror of a device buffer is the definition of plane
///   staging.
/// * `attention.masked` — the plan cache again, and `maskv:
///   In<Struct<AttnMask>>` on top: the custom `(q, kv)` mask and its own
///   CSR, published by the driver on every fire because `HasCustomMask` is
///   a folded fact. The text states that it wants the masked reading; it
///   never places the mask.
/// * `attention.kv_append` — the deepest of them, and the reason is on the
///   `untraced!` row below rather than in a signature. See
///   `WRITE_KV_TO_PAGES_ROW`.
///
/// What lands, lands because its whole input is the statement's: an
/// operand run, a weight, and numbers — and `kv_append_shared` below is
/// the one that had to argue for it.
#[kernels_macros::claims]
impl kernels::points::Attention for Ctx<'_> {
    /// THE REBASE IS THE POINT'S, AND `attn_sink_rescale` IS THE KERNEL
    /// THAT HAS IT. Two kernels in this crate fold a sink: this one, which
    /// reads the sink at `T` and multiplies the lse by `ln 2` on the way
    /// into the sigmoid, and `norm::attn_sink_correction`, which reads it
    /// at f32 and does not rebase. The declaration says base two and says
    /// the sink is the checkpoint's element, so there is exactly one of
    /// them left that answers it; the other keeps its own name for the
    /// legacy dsv4 text that still calls it and dies with that crate.
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
        attention_sink_rescale(self, o, lse, sink, Const::new(heads), Const::new(head_dim))
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
        combine_attn_outputs(
            self,
            o1,
            lse1,
            o2,
            lse2,
            o,
            lse,
            Const::new(heads),
            Const::new(head_dim),
        )
    }

    fn logit_softcap<T: kernels::points::Scalar>(
        &self,
        x: InOut<Tensor<T>>,
        cap: f32,
    ) -> Result<(), Refusal> {
        crate::attn::logit_softcap(self, x, Const::new(cap))
    }

    /// Leave dsv4's ONE plane in the pool row, as both halves of the read.
    ///
    /// THE ALIAS IS THE WHOLE BODY. `write_kv` reads `k_curr` and `v_curr`
    /// and writes `k_pages` and `v_pages` (`attn/kv_paged.cuh:202-203`),
    /// and the two source planes are `const bf16* __restrict__` that
    /// NOTHING in the kernel modifies — `restrict` constrains an object
    /// that is written through a restricted pointer, so two read-only
    /// pointers to one buffer are the legal reading of it and not a hole
    /// in it. The two DESTINATIONS stay distinct (a pool's `keys` and
    /// `values` are separate planes), which is the pair the qualifier is
    /// actually about. And the alias is the shipped arithmetic rather than
    /// a new one: the legacy dsv4 text calls
    /// `write_kv_to_pages(&kv, &kv, ..)` with the same plane in both slots
    /// (`model-legacy/src/deepseek_v4/forward/mod.rs:198`).
    ///
    /// WHY THIS LANDS WHERE [`kernels::points::Attention::kv_append`] DOES
    /// NOT. The core append's claim sits on an `untraced!` row for two
    /// reasons (see `WRITE_KV_TO_PAGES_ROW`) and the shared form answers
    /// both rather than dodging them:
    ///
    /// * `first_token` is the fire's WRITE ORIGIN — how many leading rows a
    ///   fused QKV kernel already left in the pages — and this statement has
    ///   no fused prefix to skip. dsv4 projects the plane with
    ///   `gemm.matmul`, normalises it, rotates it and appends the result;
    ///   every row of it is this point's to write, so the origin is zero
    ///   BY THE TEXT and not by a resident the driver stages.
    /// * `attn::write_kv_to_pages` is a DECLARATION STANDING FOR A CHOICE
    ///   `Boot::route` makes from the checkpoint's KV storage, and a claim
    ///   cannot delegate to a name that is not a body yet. A BODY fires with
    ///   the pool row in hand, though, and `PagedKvView::native_bf16` is
    ///   that same fact — so the choice is made here, at the fire, instead
    ///   of before the thing it reads exists. That is the plane-side branch
    ///   `.wiki/baker.md` puts inside a body, on a fact off the pool row
    ///   rather than off the operands' dims.
    ///
    /// The rest is `mla.kv_append`'s reading verbatim: the destination
    /// arithmetic is the fire's CSR, the page CSR, the last-page lengths and
    /// the row validity, and every one of those is a `PagedKvView` field.
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
        // ONE ADDRESS, TWICE — see this method's header.
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
        // THE WRITE ORIGIN IS ZERO, AND IT RIDES A POINTER. Both appenders
        // declare `first_token: In<Tensor<i32>>` and then read `ptr as i32`
        // — the mark carries a NUMBER, which is the fiction that operand
        // has always been. Null is that number's zero.
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
                // ONE BYTE PER ROW BEHIND AN `In<Tensor<i32>>`, which is the
                // fiction every appender in this file carries: the routine
                // casts the pointer to `*const u8` and the buffer must be
                // bytes. Null is legal and means every row is valid.
                In {
                    ptr: view.row_valid.cast::<i32>(),
                    rows: plane.rows,
                    width: 1,
                },
            )
        } else {
            // THE SHORTER LEG, and its operand list is this one's prefix by
            // construction (`write_kv_to_pages_bf16`'s own note says why):
            // the quantised appender has no partial rows to skip, so
            // `row_valid` is what hangs off the end and nothing else moves.
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

/// A pool row's head geometry, read off the strides it was laid out with.
///
/// A SHARED PLANE CARRIES NO HEAD GEOMETRY, which is why the declaration
/// states none: dsv4's `kv_down` is `[heads * head_dim, hidden]` and what
/// comes out of it is one rectangle with no seam in it. The POOL knows,
/// and says so twice — `driver-cuda/src/bind/views.rs::kv_view` computes
/// `seq_stride` and `head_stride` from the layer's `num_kv_heads` and
/// `head_dim`, one way round for NHD (a page is
/// `[page_size, kv_heads, head_dim]`, so a token step crosses every head
/// and a head is `head_dim`) and the other for HND (a page is
/// `[kv_heads, page_size, head_dim]`, so a token step IS `head_dim`). The
/// head width is therefore whichever of the two the layout makes it, and
/// the count is the appended row's own width over that.
///
/// READ OFF THE POOL AND NOT OFF THE TEXT, deliberately. The write has to
/// agree with the layout the pages were allocated in; a head count taken
/// from a statement and a pool laid out for another is the failure
/// `baker::geometry::agrees_with` exists to catch, and here there is
/// nothing to cross-check against because the statement says nothing.
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

#[routine(canon = "pool.boundary_decode", out(out_pos = like(positions)), out(out_req = like(positions)), out(out_rope = like(positions)))]
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

#[routine(whole, canon = "pool.boundary_prefill", out(out_pos = like(positions)), out(out_req = like(positions)), out(out_rope = like(positions)))]
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

#[routine(whole, canon = "pool.attention_lse")]
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

#[routine(bf16, canon = "index.layernorm_rope", out(idx_k = like(idx_k)))]
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

#[routine(bf16, canon = "index.rope", out(idx_q = split(idx_q, head_dim)))]
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
        /// The width of one `selection` row — `index.topk`'s stated budget.
        /// Zero when the fire carries no selection.
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
        /// `index.topk`'s answer: `[total_tokens, top_k]` of `i32`,
        /// ascending, `-1` past the end. NULL IS LEGAL and means every key —
        /// which is what the two unselected readings pass.
        pub selection: *const i32,
    }

    #[must_use]
    pub enum NaivePlan {
        Scalar { launch: Launch, head_group: i32 },
        Mma { launch: Launch },
        Declined(NaiveDecline),
    }

    /// Which arm answers, `selected` saying whether this fire carries an
    /// `index.topk` list.
    ///
    /// THE TENSOR-CORE ARM CANNOT BE SELECTED. `mla_mma_paged_kernel` stages
    /// `kBK` CONTIGUOUS keys through one `cp.async` copy of `sK`; walking a
    /// selection would mean gathering the tile, which is a different kernel
    /// and not a predicate. So a selected fire takes the scalar arm on every
    /// device, and that is the whole of the branch — no capability is read
    /// here and none is read by the two selected claims either.
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
        /// AN ARENA OFFSET IS BYTES, and this used to apply it as
        /// ELEMENTS: `base.cast::<T>().offset(o)` walks `o * size_of::<T>()`
        /// bytes, so every plan address the kernel was handed was four
        /// times as far in as `attn::plan::mla::plan` had written it.
        ///
        /// The offsets are bytes at both ends of the plan.
        /// `AlignedAllocator::alloc` is handed a SIZE IN BYTES
        /// (`alloc(4 * MAX_TOTAL_NUM_WORKS, ..)` for an i32 array of
        /// `MAX_TOTAL_NUM_WORKS`, and `rows * SIZEOF_DTYPE_O * head_dim_o`
        /// for the partial-output plane), and `Staging::put_i32s` writes at
        /// the byte offset it returns. `attn::fa2::params::offset_ptr` — the
        /// same function for the decode and prefill plans, on the path that
        /// has always fired — is `base.saturating_add(off)` on a raw
        /// address, which is this.
        ///
        /// It survived because NOTHING FIRED IT: MLA has no forward path in
        /// the legacy driver (`serve/load.rs` refuses `KvStyle::Mla` at
        /// load), so the first caller of this pack is `W7`'s columned
        /// routine and the first reader is `tests/mla_paged.rs`, where it
        /// showed up as an all-zero output — `work_indptr` read four times
        /// past its array, found zeros, and every cluster concluded it had
        /// no work.
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

/// The latent attention, once, under all four readings that name it.
///
/// NOT A ROUTINE ANY MORE, and that is `W7`'s attention half. This was
/// `#[routine(untraced)]`: a row with NO OPERAND COLUMN, an `unsafe fn`
/// taking a host `&MlaPlan` and a `MlaLayer` beside the page view, and
/// answering with a `MlaDispatch` saying which kernel it picked. A row like
/// that cannot honestly carry a `canon` — there is nothing for a binder to
/// bind and nothing for a claim to delegate to, which is exactly what
/// `kernels/src/points.rs` recorded against these four points. The four
/// `attention_mla_*_bf16` routines below are the columned form: one per
/// declared reading, each taking the statement's own operands with the plan
/// as a RAISE, and each carrying the point's name as its `canon`. This is
/// the shared body they call. It still answers with its choice, because the
/// device tests and the benchmarks read it; the routines drop it.
///
/// # The plan
///
/// `plan` must have been measured against the same page table this `Ctx`
/// will answer for: the page indices, the indptrs and the last-page lengths
/// are read as device addresses without a bound check, because the plan is
/// what bounded them. That is the `Fa2Decode` contract verbatim, and it is
/// why the four points stay CLAIM-ONLY rather than gaining bodies — see the
/// `Mla` impl's header.
///
/// `qo_indptr` arrives as a raw device address because its binder differs
/// per reading: a prefill states its CSR, a decode reads the fire's off the
/// pool row.
///
/// THE SELECTED READINGS DO NOT COME THROUGH HERE. This function's whole
/// content is the compute-capability branch, and a selection has no branch
/// to make: `MlaParams` (`attention_mla_fa2.cuh`) carries no selection and
/// the tensor-core arm cannot walk one, so `mla_naive`'s SCALAR kernel is
/// the only latent attention in this tree that honours a selection on any
/// device. [`selected_attention_mla_bf16`] is that one arm.
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
    // THE NARROWEST ARM WRITES PAST ITS OWN SHARED STORAGE, measured rather
    // than suspected. `tests/mla_paged.rs` fired arm 2 — `NUM_STAGES = 1`,
    // `CTA_TILE_KV = 16`, `QK_SHARD = false` — on an L40S (sm_89, 102 400 B
    // of smem per SM) and `compute-sanitizer` answered: *"Invalid __shared__
    // write of size 16 bytes ... Access to 0x17040 is out of bounds"*, at
    // 94 272 into a 92 672-byte allocation, from every thread of warpgroup
    // 1. The allocation is not the error: `sizeof(SharedStorage)` for this
    // arm IS 92 672 (65 536 q_nope + 8 192 q_pe + 16 384 ckv + 2 048 kpe/p
    // + 512 m_wg), which is the literal `DISPATCH_SMEM_CONFIG` compares
    // against and the number the launch passes. The overrun is 4 160 bytes
    // past `kpe_p_smem`'s base, and that union member is the one this arm
    // squeezes: it is sized `CTA_TILE_KV * max(HEAD_DIM_KPE, CTA_TILE_Q)`
    // for the KPE tile, while the P tile it shares with is written through
    // `SWIZZLE_MODE_P`, which drops to `k64B` exactly at `CTA_TILE_KV < 64`.
    //
    // THIS IS THE ONLY ARM A ≤147 967-BYTE DEVICE CAN PICK, so the refusal
    // costs nothing that worked: every latent attention on an L40S-class
    // part ends in an illegal address that kills the context, and a named
    // refusal is what the plane owes a caller instead. H100/H200 (227 KB)
    // take arm 0 and A100 (164 KB) takes arm 1; neither is implicated, and
    // neither is the sm_100 naive path above.
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

/// The `MlaLayer` staging, read off the POOL ROW a statement names.
///
/// THE LATENT POOL'S TWO PLANES ARE `keys` AND `values`, in the order
/// `driver-cuda/src/pools/mla_cache.rs` allocates and swaps them: `ckv`
/// `[pages, page_size, kv_lora_rank]` first, `kpe`
/// `[pages, page_size, qk_rope_head_dim]` second. One `PagedKvView` serves
/// both pool shapes because a page plane is a base address and a pitch, and
/// the two pitches are not the view's — they are the statement's, which is
/// why they arrive here as arguments.
fn mla_layer(kvc: &crate::views::PagedKvView, kv_lora_rank: i32, rope_dim: i32) -> MlaLayer {
    MlaLayer {
        ckv_pages: kvc.keys.cast::<c_void>(),
        kpe_pages: kvc.values.cast::<c_void>(),
        page_size: kvc.page_size,
        kv_lora_rank,
        qk_rope_head_dim: rope_dim,
    }
}

/// The rope half's width, off the query's own rectangle: `q_pe` is
/// `[tokens, heads * qk_rope_head_dim]` and `heads` is stated.
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

/// The plan raise, dereferenced, or a refusal naming it.
fn mla_plan_of(plan: &In<Struct<crate::raises::MlaPlanned>>) -> Result<&MlaPlan, Refusal> {
    if plan.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the latent attention plan this statement names",
        });
    }
    Ok(unsafe { &*plan.ptr })
}

/// The pool row, dereferenced, or a refusal naming it.
fn kv_view_of(kvc: &In<Struct<KvCache>>) -> Result<&crate::views::PagedKvView, Refusal> {
    if kvc.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the kv view this statement names",
        });
    }
    Ok(unsafe { &*kvc.ptr })
}

/// `mla.attention_decode`: one query row per request, in the latent basis.
///
/// The fire's CSR comes off the POOL ROW here and not off an operand,
/// because the declaration states none: a decode reading is one row per
/// request and the statement carries no window. `PagedKvView::qo_indptr` is
/// the same buffer `attention_mla_prefill_bf16` takes as its second operand
/// — the driver stages one per fire and hands it to both.
#[routine(no_join, canon = "mla.attention_decode")]
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
        // NOT CAUSAL: one query row per request sees the whole cached
        // prefix, and there is no second row for a mask to order it
        // against. `attention.decode`'s own dispatch reads the same way.
        false,
        kvc,
        view.qo_indptr as *const u32,
        Const::new(view.requests),
        None,
    )
    .map(|_| ())
}

/// `mla.attention_prefill`: the same reading over a query WINDOW, with the
/// statement's own CSR and the causal order it implies.
#[routine(no_join, canon = "mla.attention_prefill")]
pub fn attention_mla_prefill_bf16(
    ctx: &Ctx<'_>,
    q: In<Tensor<bf16>>,
    indptr: In<Tensor<i32>>,
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
    // The request count is the CSR operand's own row count, which is what
    // every other prefill routine in this file reads.
    let num_requests = indptr.rows;
    dispatch_attention_mla_bf16(
        ctx,
        mla_plan_of(&plan)?,
        q,
        q_pe,
        layer,
        o,
        heads,
        sm_scale,
        true,
        kvc,
        indptr.ptr as *const u32,
        Const::new(num_requests),
        None,
    )
    .map(|_| ())
}

/// The SELECTED latent attention, on the one arm that can serve it.
///
/// NO CAPABILITY BRANCH, and that is the finding rather than a shortcut.
/// [`dispatch_attention_mla_bf16`] branches because two dense kernels
/// exist; only ONE latent attention in this tree can take a selection at
/// all. The FA2 MLA kernel's `MlaParams` carries no selection pointer and
/// no budget for one, and `mla_mma_paged_kernel` stages `kBK` contiguous
/// keys through a single `cp.async` copy — a list would have to be gathered
/// into that tile, which is a different kernel. `mla_naive_paged_kernel`
/// resolves every key through the page table one at a time, so walking a
/// list costs it nothing but the indirection, and `mla_naive::plan` declines
/// the tensor-core arm whenever a selection is present.
///
/// So a selected fire runs the scalar kernel on an H100 exactly as it does
/// on a B200 — SLOWER than the arm a dense fire would take, and correct,
/// which is the trade a sparse reading is asking for in the first place.
/// The L40S-class refusal `dispatch_attention_mla_bf16` carries (the
/// `CTA_TILE_KV = 16` arm that writes past its own `SharedStorage`) does not
/// reach here at all: that is an FA2 arm and this path never picks one.
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
    // THE BUDGET IS THE OPERAND'S OWN WIDTH and is restated nowhere: the
    // selection is `[tokens, top_k]` and `index.topk` is the only place
    // `top_k` was ever written down. A statement that restated it could
    // disagree with the rectangle that was actually written.
    let sel = selection.all("the selection this attention attends over")?;
    let top_k = sel.width;
    // ONE SELECTION ROW PER QUERY ROW. The kernel reads
    // `selection + t * top_k` for every `t` the output has, so a selection
    // shorter than the query is a read past the rectangle rather than a
    // wrong answer — named here, where both rows are in hand.
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
        // The tensor-core arm is declined by `plan` whenever a selection is
        // present, so this is the scalar one and the match is exhaustive
        // rather than hopeful.
        mla_naive::MlaNaive::LaunchedScalar | mla_naive::MlaNaive::LaunchedMma => Ok(()),
    }
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

    #[routine(canon = "pool.gather")]
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

    #[routine(whole, canon = "pool.kv_append")]
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
    // THE CORE APPEND'S CLAIM SITS ON AN UNTRACED ROW, and that is the
    // whole reason `attention.kv_append` is claim-only on cuda.
    //
    // `write_kv_to_pages` is not a routine. It is a DECLARATION STANDING
    // FOR A CHOICE: `Boot::route` resolves it to `write_kv_to_pages_bf16`
    // or `write_kv_to_pages_quantised` from the KV storage the boot
    // settled, long after the trace recorded the statement. `untraced!`
    // is what says so — the row carries no operand column, which is why
    // `bind::route` sends it to `Route::Driver` and why a `#[claims]`
    // delegation would have nothing to delegate TO.
    //
    // The two legs want `first_token`, `qo_indptr` and `row_valid` beside
    // the statement's `k`, `v` and page row, and all three are runtime
    // residents the driver stages per fire (`Cx::first_token`,
    // `Cx::row_valid_d`) rather than operands any text places. A
    // declaration that stated them would be describing this plane's
    // staging; a body that conjured them would be faking it. So the point
    // keeps its default body and this row keeps its canon — which is now
    // the point's own name.
    //
    // TWO OF THOSE THREE MOVED, AND IT DOES NOT CHANGE THE ANSWER. `W7` put
    // `qo_indptr` and `row_valid` on `PagedKvView` so `mla.kv_append` could
    // be claimed by a body, and this append could read them the same way.
    // What is left is what was always decisive: `first_token` is the fire's
    // WRITE ORIGIN, a peel scalar with no home on a pool row, and the row
    // above is a DECLARATION STANDING FOR A CHOICE between two kernels that
    // `Boot::route` makes from the checkpoint's KV storage. A claim
    // delegates to a body; there is no body here to delegate to until the
    // choice has been made, and the choice is made after the trace.
    //
    // THE SHARED FORM BESIDE IT IS A BODY ALL THE SAME, and the difference
    // is the peel and only the peel. `Attention::kv_append_shared` fires
    // `write_kv_to_pages_bf16` and `write_kv_to_pages_quantised` directly,
    // branching on
    // `PagedKvView::native_bf16` — which is to say it makes `Boot::route`'s
    // choice at the fire, where the fact is, rather than at load where the
    // trace could not see it. That half of the argument above is therefore
    // about THIS ROW rather than about the point: a row with no operand
    // column cannot carry the choice, a body can. What does not move is
    // `first_token`: gptoss's fused QKV leaves a prefix of rows already in
    // the pages and the fire says how many, while dsv4's shared plane has
    // no fused producer and every row of it is the statement's. One append
    // has a write origin to be told and the other's is zero by the text.
    .canon("attention.kv_append");

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

/// NO CANON, BECAUSE NO POINT REBASES AN LSE ANY MORE. `attention.lse_ln`
/// was a point for exactly as long as the floor left the base of an lse
/// unsaid and two kernels answered it differently; `attention.decode_lse`
/// states base two now and `attention.sink` is where the one quantity that
/// is not base two — a checkpoint's logit — meets it. The legacy dsv4 text
/// still fires this launcher, and it dies with `model-dsl-legacy`.
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

#[routine(bf16, canon = "norm.res_blend", out(out = like(prefix)))]
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

#[routine(bf16, canon = "attention.logit_softcap")]
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

#[routine(bf16, canon = "mla.split_q_b")]
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

#[routine(bf16, canon = "mla.latents")]
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

#[routine(bf16, canon = "attention.merge_lse", out(o_out = like(o1)), out(lse_out = like(lse1)))]
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

/// A stated width, as a routine's `Const<i32>` asks for it.
fn width(what: &'static str, v: u32) -> Result<i32, Refusal> {
    i32::try_from(v).map_err(|_| Refusal::Wide {
        what,
        at: i64::from(v),
        max: i64::from(i32::MAX),
    })
}

/// The bf16 pin, stated as a refusal BY NAME.
///
/// Every routine the three families below delegate to is spelled at bf16 and
/// nowhere else — the absorbs because cuBLAS is handed `CUDA_R_16BF`
/// literally, the two index rotations because their `where` clauses ask for
/// `*const T: Abi`, which holds one pointee at a time. A point quantifies
/// over `Scalar`, so the claim says the pin rather than widening it with a
/// cast no kernel stands behind. The `gate.sigmoid_mul` precedent, and the
/// `ssm` conv points' before it.
fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if T::CPP == <bf16 as kernels::Elem>::CPP {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

/// The `Mla` family, claimed. Six of eleven points land.
///
/// The two cuts delegate straight through — `kimi_split_kv_a_norm` reads the
/// latent width back off the `Out` the statement sized, so the stated
/// `kv_lora_rank` is recorded and unread, the `moe.experts` reading.
///
/// THE TWO ABSORBS ARE WHY THE POINTS STATE A WIDTH THEY NEVER USE.
/// `mla_absorb_q_to_latent_bf16` is a strided batched gemm over the whole
/// `[heads, nope_dim + v_head_dim, kv_lora_rank]` bank, and the stride it
/// walks between heads is `(nope_dim + v_head_dim) * kv_lora_rank` — BOTH
/// halves, whichever half the gemm multiplies by. A `Const` weight carries
/// an address and no rectangle, so neither half is in the operands; the
/// declaration states both on both points and each body uses the one it
/// needs and passes the other through. `tokens` is NOT stated: it is
/// `q_nope.rows`, which is what the legacy lowering spliced there.
///
/// `mla.kv_append` IS THE LAUNCHER-SHAPED ONE, and it lands because the
/// pool row grew the fire it was always built out of. `write_mla` resolves
/// a destination from the query CSR, the page CSR, the last-page lengths
/// and the fire's row validity; four of those five were already fields of
/// `PagedKvView` and the two that were not (`qo_indptr`, `row_valid`) are
/// now, beside the `write_page`/`write_offset` pair that was always per-row
/// of this fire. So the whole input IS the statement's: two rectangles and
/// ONE CACHE ROW. The body delegates to `write_mla_to_pages` rather than
/// firing itself, because that launcher already exists and `norm.scale`'s
/// rule cuts the other way here — a second public name for one kernel is
/// what the delegation avoids.
///
/// `mla.latents_rope` LANDS AS TWO FIRES AND NOT A FUSION. cuda's
/// `mla_prepare_bf16` does this and three more things in one launch, which
/// is why the point was measured as a gap; but the declared statement is
/// only the cut plus a rotation of its rope half, and this plane has a
/// routine for each. `kimi_split_kv_a_norm` writes the unrotated pair and
/// `rope_partial_q_bf16` rotates `k_pe` in place, and the angles are
/// BIT-IDENTICAL to the fused kernel's: `rotate_partial` computes
/// `powf(theta, -2*dp/head_dim)` then `__sincosf`, which is `rope_cos_sin`
/// (`prelude/rope.cuh:60-68`) spelled out, and pairs `(dp, dp + half)`,
/// which is `rotate_pair_to`'s pairing. Two launches where the fusion has
/// one, and the fusion stays where it is for whoever wants it back.
///
/// `mla.absorb_q_pe` IS GONE, and that is `G2`'s answer to it rather than a
/// claim of it. The point declared an absorb whose result was
/// `[tokens, heads, kv_lora_rank + rope_dim]` — the latent with the rotated
/// half carried in its tail — and it was measured at THREE seams: the
/// absorb is a strided batched gemm with ONE activation operand, so its
/// `ldc`/`stride_c` could address the wider pitch but nothing writes the
/// tail (no per-head scatter exists); NOTHING READS ONE either, because
/// every latent attention kernel indexes `q_nope[(t * H + h) * CKV]` and
/// `q_pe[(t * H + h) * PE]` with no stride parameter between them
/// (`attention_mla_naive.cuh`, `mla_fa2::pack`'s `q_nope_stride_h` /
/// `q_pe_stride_h` off `Shape`); and the legacy glm text folded nothing —
/// it called `mla_absorbed_attention(q_nope, q_pe, ..)`, the same separate
/// pair kimi uses (`model-dsl-legacy/src/ops.rs:305-329`). A gap with no
/// producer, no consumer and no ground truth is not a gap; it is a
/// statement nobody meant. glm's text now states `mla.absorb_q` and carries
/// `q_pe` beside it, the two `_selected` readings take the pair, and the
/// declaration is deleted.
///
/// Two stay on the floor's default body, and neither is an oversight:
///
/// * `mla.attention_{decode,prefill}` — CLAIM-ONLY, and the routines below
///   are now the columned form that makes the `canon` real. What keeps them
///   from having bodies is the fa2 precedent verbatim: `attn::plan::mla`
///   measures the schedule on the HOST out of `qo_indptr`, `kv_indptr` and
///   `kv_len_arr` slices and uploads it into an int arena the launch reads.
///   A statement carries a query, a page row and three numbers; a body that
///   built the schedule would have to copy the device CSR back to the host
///   mid-fire, which is a sync a graph capture cannot record. The plane
///   stages it, as `Fa2Decode` is staged, and the points resolve through
///   `attention_mla_{decode,prefill}_bf16`.
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
        // The cut's own width is `kv_c`'s, which the statement allocated
        // from this very number; the routine reads it back off the result.
        let _ = kv_lora_rank;
        kimi_split_kv_a_norm(self, kv_a, weight, kv_c, k_pe, Const::new(eps))
    }

    /// [`kernels::points::Mla::latents`] with the rope half rotated on the
    /// way out — the cut, then the rotation, in that order and in two
    /// launches. See this impl's header for why the fused
    /// `mla_prepare_bf16` is not what answers it and why the angles agree.
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
        // `mla.latents`' reading of the same number, for the same reason.
        let _ = kv_lora_rank;
        let rope = width("the rope width this cut states", rope_dim)?;
        // READ BEFORE THE CUT CONSUMES IT: the rotation is IN PLACE over
        // the rectangle `kimi_split_kv_a_norm` is about to write, so the
        // second launch addresses the same bytes the first one left.
        let rotated = InOut {
            ptr: k_pe.ptr.cast::<bf16>(),
            rows: k_pe.rows,
            width: k_pe.width,
        };
        kimi_split_kv_a_norm(self, kv_a, weight, kv_c, k_pe, Const::new(eps))?;
        // ONE HEAD, WHOLLY ROTATED: `k_pe` is `[tokens, rope_dim]` and the
        // rope covers all of it, so the pitch and the rotated slice are the
        // same number. `rope_partial_q_bf16` passes a zero `k_width`, which
        // is what makes the kv half of `rotate_partial` empty.
        crate::rope::rope_partial_q_bf16(
            self,
            rotated,
            Const::new(rope),
            Const::new(rope),
            Const::new(theta),
            positions,
        )
    }

    /// Leave this fire's latent pair in the pool row the statement names.
    ///
    /// THE LAUNCHER FORM'S INPUTS, ALL OF THEM THE STATEMENT'S: two
    /// rectangles and one cache row. The two page planes are the pool's
    /// `keys`/`values` (see [`mla_layer`]), the two pitches are the
    /// operands' own widths, and the destination arithmetic reads the
    /// fire's CSR, the page CSR, the last-page lengths and the row validity
    /// off the same pool row — see `PagedKvView::qo_indptr` for why those
    /// last two live there.
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
            // ONE BYTE PER ROW BEHIND AN `In<Tensor<i32>>`, which is the
            // fiction every appender in this file carries: the routine
            // casts the pointer to `*const u8` and the buffer must be
            // bytes. Null is legal and means every row is valid.
            In {
                ptr: view.row_valid.cast::<i32>(),
                rows: kv_c.rows,
                width: 1,
            },
            Const::new(view.requests),
        )
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
        kimi_split_q_b(
            self,
            q_b,
            q_nope,
            q_pe,
            Const::new(width("the head count this cut states", heads)?),
            Const::new(width("the nope width this cut states", nope_dim)?),
            Const::new(width("the rope width this cut states", rope_dim)?),
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
            // The token count is the operand's own rows, which is what the
            // legacy lowering spliced into this run.
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

    /// Attend over the keys `index.topk` chose, one query row per request.
    ///
    /// A BODY AND NOT A CLAIM-ONLY ROW, and the difference is the PLAN. The
    /// two unselected readings resolve through routines because their FA2
    /// arm wants a schedule measured on the host out of three CSR slices;
    /// this one has no FA2 arm to want it. Only `mla_naive_paged_kernel`
    /// honours a selection (see [`selected_attention_mla_bf16`]) and that
    /// kernel takes no plan at all — it walks the page table itself. So the
    /// whole input IS the statement's: two query planes, a selection, one
    /// cache row and three numbers.
    ///
    /// NOT CAUSAL, for [`attention_mla_decode_bf16`]'s reason: one query row
    /// per request sees the whole cached prefix. The SELECTION is causal —
    /// `index_topk_paged` ranks only `j <= abs_q` — so the order this
    /// reading needs is already in the list it was handed.
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

    /// [`kernels::points::Mla::attention_decode_selected`] over a query
    /// WINDOW, with the statement's own CSR and the causal order it implies.
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
            // The request count is the CSR operand's own row count, which is
            // what every other prefill reading in this file reads.
            indptr.rows,
            heads,
            sm_scale,
            true,
        )
    }
}

/// The `Index` family, claimed — ALL FOUR of it, which is `G2`.
///
/// Both rotations land and both are IN PLACE, which is what the points'
/// `InOut` says and what the routines' own `out(.. = like(..))` rules
/// already said. The two that were on the floor's default body are the DSA
/// selection path, and neither was a rename away from a claim:
///
/// * `index.kv_append` — NO ROUTINE ANYWHERE ANSWERED IT, because the legacy
///   indexer never paged its keys: it scored the token plane it had just
///   written and kept nothing across fires. What answers it is `write_mla`
///   with its SECOND PLANE EMPTY — a single-plane append is the latent pair
///   minus one pitch, and the kernel's `for (i = tid; i < qk_rope_head_dim;
///   ...)` loop is already the empty loop at zero. `mla.latents_rope` uses
///   the same zero-width idiom on `rotate_partial`'s kv half.
/// * `index.topk` — a NEW KERNEL, because the old one answers a different
///   question. `dsa_index_topk_mask` scores a TOKEN-PLANE `idx_k` (the rows
///   this fire just projected), is causal only within the batch, and writes
///   a byte mask that glm's legacy text assigned to `let _index_mask` and
///   threw away. The statement names the POOL, so `index_topk_paged` scores
///   the whole cached prefix through the page table, and it answers the
///   SELECTION rather than a mask — see [`kernels::points::Index::topk`] for
///   why the list is the sizable value and the mask is not.
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
        dsa_index_knorm_rope::<bf16>(
            self,
            InOut {
                ptr: k.ptr.cast::<bf16>(),
                rows: k.rows,
                width: k.width,
            },
            Const::new(weight.v.cast::<bf16>()),
            Const::new(bias.v.cast::<bf16>()),
            Const::new(width("the rope width this norm states", rope_dim)?),
            Const::new(theta),
            Const::new(eps),
            positions,
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
        dsa_index_q_rope::<bf16>(
            self,
            InOut {
                ptr: q.ptr.cast::<bf16>(),
                rows: q.rows,
                width: q.width,
            },
            Const::new(width("the head count this rotation states", heads)?),
            Const::new(width("the head width this rotation states", head_dim)?),
            Const::new(width("the rope width this rotation states", rope_dim)?),
            Const::new(theta),
            positions,
        )
    }

    /// Leave this fire's index keys in the pool row the statement names.
    ///
    /// ONE PLANE, THROUGH THE TWO-PLANE APPEND. `write_mla` writes a latent
    /// row and a rotated row into the slot the fire's CSR resolves to, and
    /// its second write is `for (i = tid; i < qk_rope_head_dim; i +=
    /// blockDim.x)` — the empty loop at zero. So a single-plane append is
    /// that kernel with `qk_rope_head_dim = 0` and a NULL second page plane:
    /// nothing is dereferenced, so there is no aliasing question to argue
    /// (`attention.kv_append_shared` had to argue one because it writes the
    /// same rows into two live planes). `mla.latents_rope` uses the same
    /// zero-width idiom on `rotate_partial`'s kv half.
    ///
    /// THE PITCH IS READ OFF THE POOL AND CROSS-CHECKED AGAINST THE ROW,
    /// which is `attention.kv_append_shared`'s rule at a pool that has only
    /// one pitch to state. `driver-cuda/src/bind/views.rs::kv_view` sets
    /// `seq_stride` to the elements one TOKEN step crosses in a page —
    /// `kv_heads * head_dim` for NHD — and a contiguous per-token copy is
    /// correct exactly when that equals the width being appended. An HND
    /// pool is refused rather than written sideways: there a token step is
    /// `head_dim` and the row would have to be scattered per head, which is
    /// a different kernel.
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
                // NULL AND ZERO TOGETHER: the second plane is not this
                // pool's — the indexer caches a key and no value.
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
            // ONE BYTE PER ROW BEHIND AN `In<Tensor<i32>>`, which is the
            // fiction every appender in this file carries: the routine casts
            // the pointer to `*const u8` and the buffer must be bytes. Null
            // is legal and means every row is valid.
            In {
                ptr: view.row_valid.cast::<i32>(),
                rows: k.rows,
                width: 1,
            },
            Const::new(view.requests),
        )
    }

    /// Rank the whole CACHED prefix and answer the `top_k` keys that win.
    ///
    /// THE LOGITS ARE PLANE STAGING, and that is the one thing this body
    /// pulls from `self`. `index_topk_mask` keeps its scores in dynamic
    /// shared memory sized on the row count, which is affordable only
    /// because its keys are the batch; a cached prefix is not, so the scores
    /// ride a named scratch slab — `ssm::kda_qkv`'s idiom, and one fire wide,
    /// which is what the stream's serialization buys and what the tests'
    /// `FIRE` lock stands in for.
    ///
    /// THE SCRATCH'S WIDTH IS THE POOL'S OWN ANSWER TO "HOW LONG IS THE KV",
    /// which is the question the plan could not answer. `max_pages_per_request`
    /// is a per-FIRE host number on the view (`driver-cuda/src/bind/views.rs`
    /// takes it off `AttnCtx`), so `max_pages_per_request * page_size` bounds
    /// every request's `kv_len` in this fire — a host-visible bound where the
    /// per-request length is a device one. That asymmetry is exactly why the
    /// SELECTION and not a `[tokens, kv]` mask is the plan-visible value: the
    /// bound is good enough to size a scratch and not good enough to size a
    /// rectangle a text can name.
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
        // THE KEY WIDTH IS THE POOL'S TOO. The query is
        // `[tokens, heads * head_dim]` and the pool row is one key per token;
        // the same pitch reading `index.kv_append` wrote them at.
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
        // The kv bound, on the host, off the pool row. A fire whose view
        // states no page budget is one this body cannot size a scratch for,
        // and it is named rather than guessed at.
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
            .apply(Launch::per_row(out.rows.unsigned_abs(), dsa_indexer::K_BLOCK)),
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

/// One index pool row's token pitch, checked against the row being written.
///
/// See [`kernels::points::Index::kv_append`]'s body for why an HND pool is
/// refused and why the pool decides rather than the text.
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

/// The `Pool` family, claimed — and it claims NOTHING. Every point stays on
/// the floor's default body and every one of them resolves through its
/// routine's own `canon` instead, which is what claim-only means: the kernel
/// exists and fires today, and no honest delegation reaches it from a
/// statement.
///
/// THE SAME ABSENCE FIVE TIMES. DeepSeek-V4's compressed plane is three
/// resident objects beside the page table — the state halves, the running
/// scores and the absolute-position table — plus two runtime planes the fire
/// stages, `row_valid` and `request_of_token`. A statement names ONE cache
/// row and its operands, and a body cannot pull the rest from `self` because
/// they are the DRIVER's staging and not this plane's.
///
/// * `pool.boundary_decode` / `pool.boundary_prefill` — the routines write
///   THREE rectangles where the statement states two: an `out_rope` plane no
///   text in this tree reads. The declaration keeps the statement as it
///   stands rather than passing a scratch third out to swallow it — a result
///   nothing reads is not a result, and a slot no text can name has no
///   business on the floor. Both also read `row_valid`.
/// * `pool.gather` — the three residents, plus a `coff` beside the ratio.
///   The scalar alone would be derivable (it is `compressor_coff(ratio)`,
///   the driver's own rule: 4 pools 2, else 1); the residents are not.
/// * `pool.kv_append` — TWO cache views, the page table it walks and the
///   compressed pool it writes. A statement names one cache row.
/// * `pool.attention_lse` — both cache views again, and the fire's
///   request-of-token plane on top.
#[kernels_macros::claims]
impl kernels::points::Pool for Ctx<'_> {}
