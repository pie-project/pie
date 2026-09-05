//! `AttnScore`: per-key attention mass for an observation window, mirroring
//! `kernels_cuda::attn_score`.
//!
//! One entry, one stamp ladder, no workspace: recomputes the softmax weights
//! from the pages rather than materializing a `heads x window x kv_len` F32
//! slab. Unverified on device; the tests below pin the host half (ladder,
//! refusals, grid, argument order).
//!
//! [`dense`]: crate::attn::dense

use dtype::Dtype;

use crate::attn::{KvPool, PrefillPlan, RaggedTensor};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/score.metal";

/// Simdgroups per threadgroup; keys are split across them and folded once
/// per window row (the CUDA twin's `WARPS`).
const SIMDS: u32 = 8;

/// Threads per threadgroup — one Apple simdgroup is 32 lanes wide.
const THREADS: u32 = SIMDS * 32;

/// The dot-product stamps, tightest first. A stamp bounds the head width
/// from above; e.g. widths 64, 72 and 80 all share the 128 stamp.
const STAMPS: [u32; 4] = [64, 128, 256, 512];

/// The shipped point per stamp, in [`STAMPS`] order.
const CAPTURE: [&str; 4] = [
    "attn_score_capture_bfloat16_d_64",
    "attn_score_capture_bfloat16_d_128",
    "attn_score_capture_bfloat16_d_256",
    "attn_score_capture_bfloat16_d_512",
];

/// The tightest stamp that holds this head, as an index into [`STAMPS`], or
/// `None` if it is wider than the last stamp (refused, not truncated).
fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

/// Per-key attention mass for an observation window, written into a
/// caller-owned F32 slab.
///
/// `q` is the capture window's query rows paired with a window-rebased
/// `qo_indptr`; `plan`'s position table gives the causal bound; `pool` is
/// the paged cache read.
///
/// For request `r`, head `h`: row `(lane_offset + r) * plane_stride + plane
/// + h` holds `(1/rows) * sum_w softmax_j(sm_scale * <q_w, k_j>)`, where
/// `rows = min(observe, qo_len)` walks the last `rows` query rows. A
/// probability distribution over the request's live KV; no fold over heads.
///
/// The whole row is always written; the tail past live keys is exactly
/// `0.0` (the slab is reused across fires, so a stale tail would misread as
/// mass on keys that no longer exist).
///
/// # Errors
///
/// Refuses: a sliding window (semantically not this softmax, not a missing
/// instantiation); non-bf16 key pages (nothing here dequantizes); a
/// head/kv-head shape mismatch; a head wider than the widest stamp; a
/// non-F32 or wrong-width score slab; a non-`i32` position table or boundary
/// vector; `observe`, `kv_max`, `page_size`, or request count of zero; or an
/// overflowing extent. [`Error::DtypeUnsupported`] for a non-bf16 query.
///
/// A live kv extent past `kv_max` is a caller error not knowable here (it is
/// device-side); the kernel stays safe, softmax over the true extent, store
/// clamped to `kv_max`.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    lane_offset: u32,
    plane_stride: u32,
    plane: u32,
    kv_max: u32,
    requests: u32,
    scores: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.score_capture";
    if window.is_some() {
        return Err(refuse(
            OP,
            "a sliding window is stated, and a windowed row is not the softmax the \
             eviction and interpretability papers define: the mass would be a \
             distribution over the window rather than over the request's keys, and \
             every key outside it would read as unattended rather than as excluded",
        ));
    }
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    if pool.keys.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            format!(
                "the pool's key pages are stored as {:?}, and this capture reads keys \
                 straight out of the pages: it dequantizes nothing, so a quantized or \
                 fp8 pool has no scores to give",
                pool.keys.dtype
            ),
        ));
    }
    if scores.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the score slab is {:?}, and a per-key mass is an f32 rectangle",
                scores.dtype
            ),
        ));
    }
    if scores.width != kv_max {
        return Err(refuse(
            OP,
            format!(
                "the score slab's row is {} wide and the stated kv ceiling is {kv_max}; \
                 the row IS the ceiling, and a disagreement would stripe one lane's mass \
                 across another's",
                scores.width
            ),
        ));
    }
    // Unnamed seats bypass the trace-time validator, so mismatches are
    // refused here rather than asserted.
    if plan.positions.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the fire's position table is {:?}, and this capture reads the causal \
                 bound as an i32 absolute position per token",
                plan.positions.dtype
            ),
        ));
    }
    if q.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the window's boundary vector is {:?}, and this capture walks an i32 \
                 qo indptr",
                q.indptr.dtype
            ),
        ));
    }
    super::kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let Some(at) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide dot this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = super::row_heads(OP, q.data.width, head_dim)?;
    if num_q_heads % kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {kv_heads} kv heads"),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    nonzero(OP, "the slab's per-row kv ceiling", kv_max)?;
    nonzero(OP, "the requests this capture answers", requests)?;
    let page_size = u32::try_from(pool.page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| refuse(OP, "the kv page size is zero"))?;

    // The landing contract, checked only once the fire is admissible.
    debug_assert!(
        plane + num_q_heads <= plane_stride,
        "`{OP}` writes one plane per query head inside a lane's block of {plane_stride}"
    );
    debug_assert!(
        u64::from(lane_offset + requests) * u64::from(plane_stride) <= u64::from(scores.rows),
        "`{OP}` writes {requests} lanes of {plane_stride} planes from lane {lane_offset}, \
         and the slab holds {} rows",
        scores.rows
    );

    let lanes = requests.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "the grid will not launch: {requests} requests, one {THREADS}-thread group \
                 each"
            ),
        )
    })?;

    ctx.fire(
        Fire::at(FILE, CAPTURE[at]).apply(Grid::of([lanes, num_q_heads, 1], [THREADS, 1, 1])),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            plan.positions.arg(),
            scores.arg_mut(),
            stated(OP, page_size)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            stated(OP, plane_stride)?.arg(),
            stated(OP, plane)?.arg(),
            stated(OP, kv_max)?.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::probe::Probe;

    // Spelled as a number: this crate names no registry, the ceiling is
    // just a caller-stated argument.
    const KV_MAX: u32 = 2048;

    fn bf16(rows: u32, width: u32) -> Tensor {
        Tensor::new(1, rows, width, Dtype::Bf16)
    }

    fn i32s(rows: u32) -> Tensor {
        Tensor::new(2, rows, 1, Dtype::I32)
    }

    fn u32s(rows: u32) -> Tensor {
        Tensor::new(3, rows, 1, Dtype::U32)
    }

    fn slab(rows: u32) -> Tensor {
        Tensor::new(4, rows, KV_MAX, Dtype::F32)
    }

    /// A pool whose strides spell `kv_heads` heads of `head_dim`.
    fn pool(kv_heads: u32, head_dim: u32, keys: Dtype) -> KvPool {
        KvPool {
            keys: Tensor::new(5, 64, kv_heads * head_dim, keys),
            values: Tensor::new(6, 64, kv_heads * head_dim, keys),
            page_indices: u32s(8),
            page_indptr: u32s(3),
            page_size: 16,
            seq_stride: u64::from(kv_heads) * u64::from(head_dim),
            head_stride: u64::from(head_dim),
        }
    }

    fn plan(rows: u32) -> PrefillPlan {
        PrefillPlan {
            positions: i32s(rows),
            request_of_token: i32s(rows),
            mask: Tensor::new(7, rows, 1, Dtype::U8),
            mask_enabled: Tensor::new(9, rows, 1, Dtype::U8),
            mask_stride: 1,
        }
    }

    /// 16 query heads of 64 over 8 kv heads, two requests, whole slab.
    #[allow(clippy::too_many_arguments)]
    fn fire(probe: &Probe, window: Option<u32>, keys: Dtype, scores: Tensor) -> Result<(), Error> {
        let (q_heads, kv_heads, head_dim, rows) = (16u32, 8u32, 64u32, 40u32);
        capture(
            probe,
            RaggedTensor {
                data: bf16(rows, q_heads * head_dim),
                indptr: i32s(3),
            },
            &plan(rows),
            &pool(kv_heads, head_dim, keys),
            window,
            head_dim,
            kv_heads,
            0.125,
            32,
            0,
            96,
            0,
            KV_MAX,
            2,
            scores,
        )
    }

    #[test]
    fn the_head_lands_on_the_tightest_stamp_that_holds_it() {
        assert_eq!(stamp_for(40), Some(0));
        assert_eq!(stamp_for(64), Some(0));
        assert_eq!(stamp_for(65), Some(1));
        assert_eq!(stamp_for(72), Some(1));
        assert_eq!(stamp_for(128), Some(1));
        assert_eq!(stamp_for(129), Some(2));
        assert_eq!(stamp_for(256), Some(2));
        // gemma-4's global reading.
        assert_eq!(stamp_for(512), Some(3));
        // Past the last stamp is not a wider point, it is no point.
        assert_eq!(stamp_for(513), None);
        assert_eq!(STAMPS.len(), CAPTURE.len());
    }

    /// The sliding-window refusal is semantic, not a missing instantiation.
    #[test]
    fn a_sliding_window_is_refused_as_a_different_quantity() {
        let probe = Probe::default();
        let why = fire(&probe, Some(512), Dtype::Bf16, slab(4 * 96))
            .expect_err("a windowed row is not the papers' softmax");
        let said = format!("{why}");
        assert!(said.contains("sliding window"), "{said}");
        assert!(said.contains("distribution over the window"), "{said}");
        assert!(probe.fires().is_empty(), "a refused capture launched anyway");
    }

    /// The second semantic refusal: a pool this capture cannot read at all,
    /// because it reads keys straight out of the pages.
    #[test]
    fn a_quantized_key_plane_is_refused_by_name() {
        let probe = Probe::default();
        let why = fire(&probe, None, Dtype::U8, slab(4 * 96))
            .expect_err("a quantized pool has no scores to give");
        let said = format!("{why}");
        assert!(said.contains("dequantizes nothing"), "{said}");
        assert!(probe.fires().is_empty(), "a refused capture launched anyway");
    }

    #[test]
    fn a_slab_that_is_not_the_ceiling_is_refused_by_name() {
        let probe = Probe::default();
        let narrow = Tensor::new(4, 4 * 96, KV_MAX / 2, Dtype::F32);
        let why = fire(&probe, None, Dtype::Bf16, narrow)
            .expect_err("the row IS the ceiling");
        assert!(format!("{why}").contains("the row IS the ceiling"), "{why}");

        let bf16_slab = Tensor::new(4, 4 * 96, KV_MAX, Dtype::Bf16);
        let dtype = fire(&probe, None, Dtype::Bf16, bf16_slab)
            .expect_err("a per-key mass is an f32 rectangle");
        assert!(format!("{dtype}").contains("f32 rectangle"), "{dtype}");
    }

    #[test]
    fn an_observation_window_of_zero_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = capture(
            &probe,
            RaggedTensor {
                data: bf16(40, 16 * 64),
                indptr: i32s(3),
            },
            &plan(40),
            &pool(8, 64, Dtype::Bf16),
            None,
            64,
            8,
            0.125,
            0,
            0,
            96,
            0,
            KV_MAX,
            2,
            slab(4 * 96),
        )
        .expect_err("a capture that observes nothing is not a capture");
        assert!(format!("{why}").contains("observes nothing"), "{why}");
    }

    #[test]
    fn a_head_past_the_last_stamp_is_refused_by_name() {
        let probe = Probe::default();
        let why = capture(
            &probe,
            RaggedTensor {
                data: bf16(40, 2 * 1024),
                indptr: i32s(3),
            },
            &plan(40),
            &pool(2, 1024, Dtype::Bf16),
            None,
            1024,
            2,
            0.125,
            32,
            0,
            96,
            0,
            KV_MAX,
            2,
            slab(4 * 96),
        )
        .expect_err("a 1024-wide head is past the ladder");
        assert!(format!("{why}").contains("stamped for"), "{why}");
    }

    #[test]
    fn an_element_this_plane_has_no_point_for_is_refused_by_dtype() {
        let probe = Probe::default();
        let why = capture(
            &probe,
            RaggedTensor {
                data: Tensor::new(1, 40, 16 * 64, Dtype::F32),
                indptr: i32s(3),
            },
            &plan(40),
            &pool(8, 64, Dtype::Bf16),
            None,
            64,
            8,
            0.125,
            32,
            0,
            96,
            0,
            KV_MAX,
            2,
            slab(4 * 96),
        )
        .expect_err("this capture is stamped for bf16 alone");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }
}
