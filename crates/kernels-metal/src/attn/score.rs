use dtype::Dtype;

use crate::attn::{KvPool, PrefillPlan, RaggedTensor};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/score.metal";

const SIMDS: u32 = 8;

const THREADS: u32 = SIMDS * 32;

const STAMPS: [u32; 4] = [64, 128, 256, 512];

const CAPTURE: [&str; 4] = [
    "attn_score_capture_bfloat16_d_64",
    "attn_score_capture_bfloat16_d_128",
    "attn_score_capture_bfloat16_d_256",
    "attn_score_capture_bfloat16_d_512",
];

fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    kv_max: u32,
    requests: u32,
    scores: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.score_capture";
    if super::q8::enabled(pool, head_dim) {
        return Err(refuse(
            OP,
            "score capture is not yet supported with `kernel.sdpa.kv_q8` on",
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
                "the score rows are {:?}, and a per-key mass is an f32 rectangle",
                scores.dtype
            ),
        ));
    }
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
    nonzero(OP, "the per-head kv ceiling", kv_max)?;
    nonzero(OP, "the requests this capture answers", requests)?;
    let page_size = u32::try_from(pool.page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| refuse(OP, "the kv page size is zero"))?;

    if scores.width != num_q_heads.saturating_mul(kv_max) || scores.rows < requests {
        return Err(refuse(
            OP,
            format!(
                "the score rows are {} by {}, and {requests} lanes capture {num_q_heads} \
                 heads of {kv_max} keys",
                scores.rows, scores.width
            ),
        ));
    }

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
            stated(OP, kv_max)?.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::probe::Probe;

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

    fn rows(heads: u32) -> Tensor {
        Tensor::new(4, 2, heads * KV_MAX, Dtype::F32)
    }

    fn pool(kv_heads: u32, head_dim: u32, keys: Dtype) -> KvPool {
        KvPool {
            keys: Tensor::new(5, 64, kv_heads * head_dim, keys),
            values: Tensor::new(6, 64, kv_heads * head_dim, keys),
            page_indices: u32s(8),
            page_indptr: u32s(3),
            page_size: 16,
            seq_stride: u64::from(kv_heads) * u64::from(head_dim),
            head_stride: u64::from(head_dim),
            q8: false,
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

    fn fire(
        probe: &Probe,
        q: Tensor,
        kv_heads: u32,
        head_dim: u32,
        keys: Dtype,
        observe: u32,
        scores: Tensor,
    ) -> Result<(), Error> {
        capture(
            probe,
            RaggedTensor {
                data: q,
                indptr: i32s(3),
            },
            &plan(40),
            &pool(kv_heads, head_dim, keys),
            head_dim,
            kv_heads,
            0.125,
            observe,
            KV_MAX,
            2,
            scores,
        )
    }

    #[test]
    fn score_every_case() {
        the_head_lands_on_the_tightest_stamp_that_holds_it();
        a_quantized_key_plane_is_refused_by_name();
        rows_that_are_not_heads_of_the_ceiling_are_refused();
        an_observation_window_of_zero_rows_is_refused_by_name();
        a_head_past_the_last_stamp_is_refused_by_name();
        an_element_this_plane_has_no_point_for_is_refused_by_dtype();
    }

    fn the_head_lands_on_the_tightest_stamp_that_holds_it() {
        assert_eq!(stamp_for(40), Some(0));
        assert_eq!(stamp_for(64), Some(0));
        assert_eq!(stamp_for(65), Some(1));
        assert_eq!(stamp_for(128), Some(1));
        assert_eq!(stamp_for(256), Some(2));
        assert_eq!(stamp_for(512), Some(3));
        assert_eq!(stamp_for(513), None);
        assert_eq!(STAMPS.len(), CAPTURE.len());
    }

    fn a_quantized_key_plane_is_refused_by_name() {
        let probe = Probe::default();
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::U8, 32, rows(16))
            .expect_err("a quantized pool has no scores to give");
        assert!(format!("{why}").contains("dequantizes nothing"), "{why}");
        assert!(
            probe.fires().is_empty(),
            "a refused capture launched anyway"
        );
    }

    fn rows_that_are_not_heads_of_the_ceiling_are_refused() {
        let probe = Probe::default();
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, 32, rows(8))
            .expect_err("a row holds every head's keys");
        assert!(format!("{why}").contains("capture 16 heads"), "{why}");

        let bf16_rows = Tensor::new(4, 2, 16 * KV_MAX, Dtype::Bf16);
        let dtype = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, 32, bf16_rows)
            .expect_err("a per-key mass is an f32 rectangle");
        assert!(format!("{dtype}").contains("f32 rectangle"), "{dtype}");
    }

    fn an_observation_window_of_zero_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, 0, rows(16))
            .expect_err("a capture that observes nothing is not a capture");
        assert!(format!("{why}").contains("observes nothing"), "{why}");
    }

    fn a_head_past_the_last_stamp_is_refused_by_name() {
        let probe = Probe::default();
        let why = fire(
            &probe,
            bf16(40, 2 * 1024),
            2,
            1024,
            Dtype::Bf16,
            32,
            rows(2),
        )
        .expect_err("a 1024-wide head is past the ladder");
        assert!(format!("{why}").contains("stamped for"), "{why}");
    }

    fn an_element_this_plane_has_no_point_for_is_refused_by_dtype() {
        let probe = Probe::default();
        let q = Tensor::new(1, 40, 16 * 64, Dtype::F32);
        let why = fire(&probe, q, 8, 64, Dtype::Bf16, 32, rows(16))
            .expect_err("this capture is stamped for bf16 alone");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }
}
