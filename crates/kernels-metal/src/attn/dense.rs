//! `Dense`: bidirectional attention over the patch window — the vision
//! towers' one real kernel, mirroring `kernels_cuda::attn_dense`. Shares
//! nothing with the paged family (no kv pool, no plan, no mask ladder); one
//! entry, one stamp ladder, no workspace (online softmax needs no scratch).
//! Unverified on device: what the tests below pin is the host half only.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "attn/dense.metal";

/// Simdgroups per threadgroup. Keys are split across them and folded once at
/// the end, so this is the kernel's only parallelism knob above the head. The
/// shader bakes it into its instantiations; the two agree here or nowhere.
const SIMDS: u32 = 4;

/// Threads per threadgroup — one Apple simdgroup is 32 lanes wide.
const THREADS: u32 = SIMDS * 32;

/// The accumulator stamps, tightest first. A stamp is register footprint and
/// threadgroup allocation, not a shape: the live head width may be anything
/// at or below it. Ladder shared with the CUDA twin; a head past the last
/// stamp is refused by name rather than truncated.
const STAMPS: [u32; 3] = [64, 128, 256];

/// The shipped point per stamp, in [`STAMPS`] order.
const DENSE: [&str; 3] = [
    "dense_bidirectional_bfloat16_d_64",
    "dense_bidirectional_bfloat16_d_128",
    "dense_bidirectional_bfloat16_d_256",
];

/// The tightest stamp that holds this head, as an index into [`STAMPS`] —
/// or nothing, because a head wider than the last stamp is refused rather
/// than silently truncated.
fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

/// The head count a row's width spells at a stated head width.
fn row_heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// The image count this patch window's indptr spells.
///
/// The segment list is a fire table the shell assembles, not a value any op
/// names, so the trace-time validator never sees it — which is why its dtype
/// and its length are refused here rather than asserted (the boundary rule at
/// [`refuse`](crate::encode::refuse)).
fn images_of(op: &'static str, segments: Tensor) -> Result<i32, Error> {
    if segments.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the patch window's segment list is {:?}, and this attention walks an i32 \
                 boundary vector",
                segments.dtype
            ),
        ));
    }
    let images = segments.rows.saturating_sub(1);
    if images == 0 {
        return Err(refuse(op, "the patch window's segment list spells no images"));
    }
    stated(op, images)
}

/// Bidirectional dense attention, block-diagonal per image. `q`, `k`, `v` are
/// patch rows (`[patch_rows, heads * head_dim]`, bf16); `o` lands one row per
/// query row. `segments` is the patch axis's indptr (`i32`, `[images + 1]`):
/// row `n` attends to the rows of the image whose span contains it, both
/// directions. Grouped heads supported by reading, not expanding.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a row
/// width that does not divide by the stated head width, a head wider than the
/// widest stamp, query heads that do not divide by kv heads, a segment list
/// that is not an `i32` indptr of at least one image, or a grid that will not
/// launch.
#[allow(clippy::too_many_arguments)]
pub fn bidirectional(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dense";
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    nonzero(OP, "the head width this attention states", head_dim)?;
    let Some(at) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide accumulator this kernel \
                 is stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => DENSE[at] });
    let num_q_heads = row_heads(OP, "query", q.width, head_dim)?;
    let num_kv_heads = row_heads(OP, "key", k.width, head_dim)?;
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    // The named refusals above judge the plan-visible geometry; these two are
    // the landing contract, checked only once the plan itself is admissible.
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(
        k.rows, v.rows,
        "`{OP}` reads one value row per key row of the patch window"
    );

    let images = images_of(OP, segments)?;
    let rows = nonzero(OP, "the patch rows this attention answers", q.rows)?;
    let lanes = num_q_heads.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "the grid will not launch: {num_q_heads} query heads, one {THREADS}-thread \
                 group each"
            ),
        )
    })?;

    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([lanes, rows, 1], [THREADS, 1, 1])),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg_mut(),
            segments.arg(),
            images.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::probe::Probe;

    fn bf16(rows: u32, width: u32) -> Tensor {
        Tensor::new(1, rows, width, Dtype::Bf16)
    }

    fn indptr(images: u32) -> Tensor {
        Tensor::new(2, images + 1, 1, Dtype::I32)
    }

    /// The one property a stamp ladder has to have: the head lands on the
    /// tightest point that holds it, and the widths the CUDA goldens
    /// exercise all land somewhere.
    #[test]
    fn the_head_lands_on_the_tightest_stamp_that_holds_it() {
        assert_eq!(stamp_for(40), Some(0));
        assert_eq!(stamp_for(64), Some(0));
        assert_eq!(stamp_for(65), Some(1));
        assert_eq!(stamp_for(72), Some(1));
        assert_eq!(stamp_for(80), Some(1));
        assert_eq!(stamp_for(128), Some(1));
        assert_eq!(stamp_for(129), Some(2));
        assert_eq!(stamp_for(256), Some(2));
        // Past the last stamp is not a wider point, it is no point.
        assert_eq!(stamp_for(257), None);
        assert_eq!(STAMPS.len(), DENSE.len());
    }

    #[test]
    fn a_head_past_the_last_stamp_is_refused_by_name() {
        let probe = Probe::default();
        let why = bidirectional(
            &probe,
            bf16(8, 512),
            bf16(8, 512),
            bf16(8, 512),
            indptr(1),
            512,
            0.125,
            bf16(8, 512),
        )
        .expect_err("a 512-wide head is past the ladder");
        assert!(format!("{why}").contains("stamped for"), "{why}");
        assert!(probe.fires().is_empty(), "a refused plan launched anyway");
    }

    #[test]
    fn a_row_that_is_no_whole_number_of_heads_is_refused_by_name() {
        let probe = Probe::default();
        let why = bidirectional(
            &probe,
            bf16(8, 2 * 64 + 1),
            bf16(8, 2 * 64),
            bf16(8, 2 * 64),
            indptr(1),
            64,
            0.125,
            bf16(8, 2 * 64 + 1),
        )
        .expect_err("a ragged query row has no head count");
        assert!(format!("{why}").contains("query"), "{why}");
    }

    #[test]
    fn query_heads_that_do_not_group_are_refused_by_name() {
        let probe = Probe::default();
        let why = bidirectional(
            &probe,
            bf16(8, 6 * 64),
            bf16(8, 4 * 64),
            bf16(8, 4 * 64),
            indptr(1),
            64,
            0.125,
            bf16(8, 6 * 64),
        )
        .expect_err("six heads do not group over four");
        assert!(format!("{why}").contains("group over"), "{why}");
    }

    #[test]
    fn a_segment_list_that_is_not_an_indptr_is_refused_by_name() {
        let probe = Probe::default();
        let wrong_dtype = bidirectional(
            &probe,
            bf16(8, 64),
            bf16(8, 64),
            bf16(8, 64),
            Tensor::new(2, 2, 1, Dtype::U32),
            64,
            0.125,
            bf16(8, 64),
        )
        .expect_err("a u32 boundary vector is not this plane's indptr");
        assert!(format!("{wrong_dtype}").contains("i32"), "{wrong_dtype}");

        let no_images = bidirectional(
            &probe,
            bf16(8, 64),
            bf16(8, 64),
            bf16(8, 64),
            indptr(0),
            64,
            0.125,
            bf16(8, 64),
        )
        .expect_err("a one-entry indptr spells no image");
        assert!(format!("{no_images}").contains("no images"), "{no_images}");
    }

    #[test]
    fn an_element_this_plane_has_no_point_for_is_refused_by_dtype() {
        let probe = Probe::default();
        let why = bidirectional(
            &probe,
            Tensor::new(1, 8, 64, Dtype::F32),
            Tensor::new(1, 8, 64, Dtype::F32),
            Tensor::new(1, 8, 64, Dtype::F32),
            indptr(1),
            64,
            0.125,
            Tensor::new(1, 8, 64, Dtype::F32),
        )
        .expect_err("this attention is stamped for bf16 alone");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }
}
