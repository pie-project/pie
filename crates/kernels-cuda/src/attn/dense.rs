//! `Dense`: bidirectional attention over the patch window — the vision
//! towers' one real kernel (`.wiki/alto/multimodal.md` §2, ownership decided
//! 2026-08-30).
//!
//! **A NEW FILE ON PURPOSE.** The paged family next door is the human's live
//! area, and this op shares nothing with it but the word *attention*: no kv
//! pool, no page tables, no append, no plan, no mask ladder, no log-sum-exp
//! plane. Everything it needs is q, k, v, and the patch axis's own indptr —
//! so it is a file of its own rather than an arm inside [`attn`](crate::attn),
//! and the two can be read, changed and broken independently.
//!
//! **Where the module hangs, and why it is not `attn::dense`.** The design
//! spells the op `attn::dense` and the FILE sits where it says
//! (`src/attn/dense.rs`), but a child module can only be declared by its
//! parent, and `src/attn.rs` is a file this wave may not touch. `lib.rs`
//! re-homes the declaration with `#[path]` instead, so the module path is
//! `kernels_cuda::attn_dense` until that seam reopens and the one line
//! `pub mod dense;` can move into `attn.rs` where it belongs. Nothing else
//! about the op changes when it does.
//!
//! One entry, one instantiation ladder, no workspace: see the unit's header
//! for the online-softmax argument that makes the absence of scratch a
//! property rather than an oversight.

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

/// **BIDIRECTIONAL DENSE ATTENTION, BLOCK-DIAGONAL PER IMAGE.**
///
/// `q`, `k` and `v` are patch rows — `[patch_rows, heads * head_dim]`, bf16 —
/// and `o` lands one row per query row at q's own shape. `segments` is the
/// patch axis's indptr (`i32`, `[images + 1]`): row `n` attends to the rows
/// of the image whose span contains it, in both directions, and to nothing
/// else. A row past the last span is a rung's padding and lands zeros.
///
/// `sm_scale` is the caller's, as everywhere else in this plane — the towers
/// state `1 / sqrt(head_dim)` and a checkpoint that states something else is
/// obeyed rather than second-guessed.
///
/// Grouped heads are supported by reading, not expanding: `k`/`v` may spell
/// fewer heads than `q` as long as the counts divide.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a row
/// width that does not divide by the stated head width, a head wider than
/// the widest stamp, query heads that do not divide by kv heads, a segment
/// list that is not an `i32` indptr of at least one image, or an output
/// rectangle that is not q's.
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

    let images = kv::lanes_of(OP, segments)?;
    let rows = count(OP, "the patch rows this attention answers", q.rows)?;

    // q's row, plus one accumulator plane and two folding words per warp.
    // The whole workspace, and it is shared memory — the fire path allocates
    // nothing (design Article 7).
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
