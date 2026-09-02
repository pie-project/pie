//! `AttnScore`: per-key attention mass for an observation window. Shares the
//! pages with the fa2 family and nothing else — no schedule to walk, no
//! split-kv partials, no merge, no log-sum-exp plane, no `o`; borrows the
//! plan only for its shape triple. One entry, one instantiation ladder, no
//! workspace: recomputes softmax weights straight out of the pages rather
//! than materializing a `heads x window x kv_len` F32 slab per fire.

use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/score.cuh";

/// Warps per block. Keys are split across them and folded once per window
/// row, so this is the kernel's only parallelism knob above the head.
const WARPS: u32 = 8;

const BLOCK: u32 = WARPS * 32;

/// The dot-product stamps, tightest first. A stamp is the unrolled per-lane
/// length (`stamp / 32` elements), so head widths at or below it share it
/// unpadded.
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

/// Per-key attention mass for an observation window, into a caller-owned F32
/// slab.
///
/// `q` is the capture window's query rows (`[rows, num_q_heads * head_dim]`,
/// bf16) paired with the window-rebased `qo_indptr` (`i32`, `[requests + 1]`).
/// `plan` is read only for the shape triple it was carved at; `pool` is the
/// paged cache this layer read.
///
/// For request `r` and query head `h` the output row is
/// `scores.ptr + ((lane_offset + r) * plane_stride + plane + h) * kv_max`
/// floats, and it holds
///
/// ```text
///   out[j] = (1 / rows) * sum over w of softmax_j( sm_scale * <q_w, k_j> )
/// ```
///
/// where `rows = min(observe, qo_len)`, `w` walks the request's last `rows`
/// query rows, and each row's softmax is taken over its own causal limit
/// `min(kv_len - rows + w + 1, kv_len)`. Result sums to one over `[0, kv_len)`:
/// TOVA's number at `observe = 1`, SnapKV's at `observe = 32`.
///
/// The whole row is written, always: `[kv_len, kv_max)` lands exactly `0.0`
/// on every path, including degenerate ones, since the slab is reused across
/// fires and a stale tail would otherwise be mistaken for zero mass.
///
/// # Errors
///
/// A refusal when:
///
/// - the plan was not carved at this head width, kv head count or window
///   ([`PrefillPlan::accepts`]);
/// - a sliding window is stated at all — not the softmax eviction and
///   interpretability papers define;
/// - the pool's key pages are not bf16 storage (this capture dequantizes
///   nothing);
/// - the slab is not F32, or its row is not `kv_max` wide;
/// - the query row width does not divide by the stated head width;
/// - the query heads do not group over the kv heads;
/// - the head is wider than the widest stamp;
/// - `observe` is zero;
/// - `kv_max` is zero, or any stated extent overflows the kernel's `int`.
///
/// [`Error::DtypeUnsupported`] for a query in anything but bf16.
///
/// `kv_len > kv_max` is a caller error refused upstream, not knowable here
/// (`kv_len` is device-side): the softmax is still taken over the true
/// extent and only the store is clamped to `kv_max`.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx,
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
    scores: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.score_capture";
    plan.accepts(OP, head_dim, Some(kv_heads), window)?;
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
    if !kv::native_bf16(pool) {
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
    let head_dim = count(OP, "the head width this capture states", head_dim)?;
    let head_width = head_dim.unsigned_abs();
    let Some(stamp) = stamp_for(head_width) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_width}-wide head is wider than the {}-wide dot this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads(OP, "query", q.data.width, head_width)?;
    let num_kv_heads = count(OP, "the kv head count this capture states", kv_heads)?.unsigned_abs();
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    let kv_ceiling = count(OP, "the slab's per-row kv ceiling", kv_max)?;
    let page_size = count(OP, "the pool's page size", pool.page_size.unsigned_abs())?;

    let requests = kv::lanes_of(OP, q.indptr)?.unsigned_abs();
    debug_assert_eq!(
        num_q_heads, plan.shape.num_q_heads,
        "`{OP}` captures one row per query head of the plan it was carved with"
    );
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

    // The q row plus the block's two fold words per warp — the whole
    // workspace, in shared memory (the fire path allocates nothing).
    let floats = head_width.saturating_add(2 * WARPS);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("a {head_width}-wide head over {WARPS} warps overflows its shared plane"),
            )
        })?;

    let hnd = if pool.layout == 0 {
        "::pie::false_type::value"
    } else {
        "::pie::true_type::value"
    };

    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::attn::score_capture<{stamp}, {WARPS}, {hnd}>"
            )),
        )
        .apply(Launch::grid([requests, num_q_heads, 1], [BLOCK, 1, 1]).smem(smem)),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            scores.arg(),
            page_size.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            stated(OP, plane_stride)?.arg(),
            stated(OP, plane)?.arg(),
            kv_ceiling.arg(),
        ],
    )
}
