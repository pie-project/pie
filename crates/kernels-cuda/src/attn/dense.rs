//! `Dense`: bidirectional attention over the patch window, the vision towers' one real kernel. Shares nothing with the paged attention family: no kv pool, page tables, append, plan, mask ladder or log-sum-exp plane — just q, k, v and the patch axis's own indptr.
//! Module path is `kernels_cuda::attn_dense` via `#[path]` in `lib.rs`, standing in for `attn::dense` until `src/attn.rs` can declare it directly.

use crate::attn::kv;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "attn/dense.cuh";

/// Warps per block. Keys are split across them and folded once at the end,
/// so this is the kernel's only parallelism knob above the head.
const WARPS: u32 = 4;

const BLOCK: u32 = WARPS * 32;

/// The accumulator stamps, tightest first. A stamp is register footprint
/// (`stamp / 32` floats per lane) and not a shape: the live head width may be
/// anything at or below it, which is what lets 64, 72 and 80 share the
/// 128-wide stamp without any of them being padded.
const STAMPS: [u32; 3] = [64, 128, 256];

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

/// The tightest stamp that holds this head, or nothing — a head wider than
/// the last stamp is refused rather than silently truncated.
fn stamp_for(head_dim: u32) -> Option<u32> {
    STAMPS.into_iter().find(|stamp| head_dim <= *stamp)
}

/// Bidirectional dense attention, block-diagonal per image.
///
/// `q`, `k`, `v` are patch rows (`[patch_rows, heads * head_dim]`, bf16); `o` lands one row per query row at q's own shape. `segments` is the patch axis's indptr (`i32`, `[images + 1]`): row `n` attends both ways to the rows of the image whose span contains it; a row past the last span lands zeros.
/// `sm_scale` is the caller's, unvalidated. Grouped heads: `k`/`v` may spell fewer heads than `q` as long as the counts divide.
///
/// Errs [`Error::DtypeUnsupported`] for anything but bf16, or a refusal for a mismatched row/head width, ungrouped heads, a bad segment list, or an output rectangle that isn't q's.
pub fn bidirectional(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dense";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(k.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    debug_assert_eq!(v.dtype, q.dtype, "`{OP}` reads q, k and v in one element");
    let head_dim = count(OP, "the head width this attention states", head_dim)?;
    let head_width = head_dim.unsigned_abs();
    let Some(stamp) = stamp_for(head_width) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_width}-wide head is wider than the {}-wide accumulator this kernel \
                 is stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads(OP, "query", q.width, head_width)?;
    let num_kv_heads = row_heads(OP, "key", k.width, head_width)?;
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!(
                "{num_q_heads} query heads do not group over {num_kv_heads} kv heads"
            ),
        ));
    }
    // the landing contract, checked only once the plan itself is admissible.
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "`{OP}` lands one output row per query row"
    );
    debug_assert_eq!(
        k.rows, v.rows,
        "`{OP}` reads one value row per key row of the patch window"
    );

    let images = kv::lanes_of(OP, segments)?;
    let rows = count(OP, "the patch rows this attention answers", q.rows)?;

    // q's row, plus one accumulator plane and two folding words per warp; the whole workspace, all shared memory.
    let floats = head_width.saturating_mul(WARPS + 1).saturating_add(2 * WARPS);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("a {head_width}-wide head over {WARPS} warps overflows its shared plane"),
            )
        })?;

    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::attn::dense_bidirectional<{stamp}, {WARPS}>")),
        )
        .apply(
            Launch::grid([rows.unsigned_abs(), num_q_heads, 1], [BLOCK, 1, 1]).smem(smem),
        ),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            o.arg(),
            segments.arg(),
            images.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
        ],
    )
}
