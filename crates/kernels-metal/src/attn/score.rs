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

const PAGES: &str = "attn_score_pages";

/// Where a fire's capturing lanes lay their score rows: `store`, the
/// buffer the rows lie in (bound from its start), `rows`, each fire lane's
/// row's first element in it (a u64; `u64::MAX` for a lane that captures
/// nothing), and `observe`, the query rows each fire lane's program
/// observes (a u32, its last ones). This window's first lane is `first`.
#[derive(Clone, Copy, Debug)]
pub struct CaptureRows {
    pub store: Tensor,
    pub rows: Tensor,
    pub observe: Tensor,
    pub first: u32,
}

/// Each capturing request's attention mass a key of its first `kv_max`
/// (the load's context), averaged over the query rows its program observes
/// ([`CaptureRows::observe`]) and
/// over the model's `total_heads` query heads (this launch attends its
/// `q` heads of them), added to its row ([`CaptureRows`]); with `zero`, the
/// row is cleared first.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    kv_max: u32,
    total_heads: u32,
    zero: bool,
    requests: u32,
    rows: CaptureRows,
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
    if rows.store.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the score rows are {:?}, and a per-key mass is an f32 rectangle",
                rows.store.dtype
            ),
        ));
    }
    if rows.rows.dtype != Dtype::U64 {
        return Err(refuse(
            OP,
            format!(
                "the capturing lanes' row table is {:?}, and each start is a u64 element \
                 offset",
                rows.rows.dtype
            ),
        ));
    }
    if plan.last_page_len.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the plan's last-page lengths are {:?}, and this capture reads each \
                 request's kv length off an i32 one",
                plan.last_page_len.dtype
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
    if rows.observe.dtype != Dtype::U32 || rows.observe.rows < rows.rows.rows {
        return Err(refuse(
            OP,
            format!(
                "the capturing lanes' observe table is {} {:?}, and each fire lane's rows are \
                 a u32 beside its row's start",
                rows.observe.rows, rows.observe.dtype
            ),
        ));
    }
    nonzero(OP, "the keys a row holds", kv_max)?;
    if total_heads < num_q_heads {
        return Err(refuse(
            OP,
            format!(
                "this launch attends {num_q_heads} query heads of a model stated to have \
                 {total_heads}"
            ),
        ));
    }
    nonzero(OP, "the requests this capture answers", requests)?;
    let page_size = u32::try_from(pool.page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| refuse(OP, "the kv page size is zero"))?;

    if rows.rows.rows < rows.first.saturating_add(requests) {
        return Err(refuse(
            OP,
            format!(
                "the row table holds {} capturing lanes, and this window's {requests} start at \
                 {}",
                rows.rows.rows, rows.first
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
        Fire::at(FILE, CAPTURE[at]).apply(Grid::of([lanes, 1, 1], [THREADS, 1, 1])),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            plan.last_page_len.arg(),
            rows.store.arg_mut(),
            stated(OP, page_size)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
            rows.observe.arg(),
            stated(OP, kv_max)?.arg(),
            pool.page_stride[0].arg(),
            pool.page_stride[1].arg(),
            crate::tensor::PLANE_AT_BINDING.arg(),
            crate::tensor::PLANE_AT_BINDING.arg(),
            rows.rows.arg(),
            rows.first.arg(),
            stated(OP, total_heads)?.arg(),
            u32::from(zero).arg(),
        ],
    )
}

/// Each capturing request's key row (`keys`' starts, `keys_len` keys)
/// summed a page of `page_size` into its page row (`pages`' starts, the
/// pages the keys come to), for `requests` requests from the window's
/// first.
#[allow(clippy::too_many_arguments)]
pub fn pages(
    ctx: &Ctx<'_>,
    store: Tensor,
    keys: Tensor,
    pages: Tensor,
    first: u32,
    keys_len: u32,
    page_size: u32,
    requests: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.score_pages";
    if store.dtype != Dtype::F32 || keys.dtype != Dtype::U64 || pages.dtype != Dtype::U64 {
        return Err(refuse(
            OP,
            "the rows are f32 and each row's start a u64 element offset",
        ));
    }
    nonzero(OP, "the keys a row holds", keys_len)?;
    nonzero(OP, "the keys a page holds", page_size)?;
    nonzero(OP, "the requests this sum answers", requests)?;
    let page_count = keys_len.div_ceil(page_size);
    if keys.rows < first.saturating_add(requests) || pages.rows < first.saturating_add(requests) {
        return Err(refuse(
            OP,
            format!(
                "the row tables hold {} and {} capturing lanes, and this window's {requests} \
                 start at {first}",
                keys.rows, pages.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at(FILE, PAGES).apply(Grid::of([page_count, requests, 1], [64.min(page_count), 1, 1])),
        &[
            store.arg_mut(),
            keys.arg(),
            pages.arg(),
            first.arg(),
            keys_len.arg(),
            page_size.arg(),
            page_count.arg(),
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

    /// The rows of `lanes` capturing lanes, from the window's first.
    fn rows(lanes: u32) -> CaptureRows {
        CaptureRows {
            store: Tensor::new(4, 1, 1, Dtype::F32),
            rows: Tensor::new(10, lanes, 1, Dtype::U64),
            observe: Tensor::new(11, lanes, 1, Dtype::U32),
            first: 0,
        }
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
            page_stride: [0; 2],
        }
        .paged_densely()
    }

    fn plan(rows: u32) -> PrefillPlan {
        PrefillPlan {
            positions: i32s(rows),
            last_page_len: i32s(2),
            request_of_token: i32s(rows),
            mask: Tensor::new(7, rows, 1, Dtype::U8),
            mask_enabled: Tensor::new(9, rows, 1, Dtype::U32),
            mask_stride: 1,
        }
    }

    fn fire(
        probe: &Probe,
        q: Tensor,
        kv_heads: u32,
        head_dim: u32,
        keys: Dtype,
        scores: CaptureRows,
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
            KV_MAX,
            16,
            true,
            2,
            scores,
        )
    }

    #[test]
    fn score_every_case() {
        the_head_lands_on_the_tightest_stamp_that_holds_it();
        a_quantized_key_plane_is_refused_by_name();
        a_row_table_short_of_the_windows_lanes_is_refused();
        an_observe_table_short_of_the_row_table_is_refused_by_name();
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
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::U8, rows(2))
            .expect_err("a quantized pool has no scores to give");
        assert!(format!("{why}").contains("dequantizes nothing"), "{why}");
        assert!(
            probe.fires().is_empty(),
            "a refused capture launched anyway"
        );
    }

    fn a_row_table_short_of_the_windows_lanes_is_refused() {
        let probe = Probe::default();
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, rows(1))
            .expect_err("each of the window's two capturing lanes has its row's start");
        assert!(format!("{why}").contains("the row table holds 1"), "{why}");

        let bf16_rows = CaptureRows {
            store: Tensor::new(4, 1, 1, Dtype::Bf16),
            ..rows(2)
        };
        let dtype = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, bf16_rows)
            .expect_err("a per-key mass is an f32 rectangle");
        assert!(format!("{dtype}").contains("f32 rectangle"), "{dtype}");
    }

    fn an_observe_table_short_of_the_row_table_is_refused_by_name() {
        let probe = Probe::default();
        let short = CaptureRows {
            observe: Tensor::new(11, 1, 1, Dtype::U32),
            ..rows(2)
        };
        let why = fire(&probe, bf16(40, 16 * 64), 8, 64, Dtype::Bf16, short)
            .expect_err("each fire lane states the rows it observes");
        assert!(format!("{why}").contains("observe table"), "{why}");
    }

    fn a_head_past_the_last_stamp_is_refused_by_name() {
        let probe = Probe::default();
        let why = fire(
            &probe,
            bf16(40, 2 * 1024),
            2,
            1024,
            Dtype::Bf16,
            rows(2),
        )
        .expect_err("a 1024-wide head is past the ladder");
        assert!(format!("{why}").contains("stamped for"), "{why}");
    }

    fn an_element_this_plane_has_no_point_for_is_refused_by_dtype() {
        let probe = Probe::default();
        let q = Tensor::new(1, 40, 16 * 64, Dtype::F32);
        let why = fire(&probe, q, 8, 64, Dtype::Bf16, rows(2))
            .expect_err("this capture is stamped for bf16 alone");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }
}
