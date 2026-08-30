//! `AttnScore`: per-key attention mass for an observation window — the alto
//! observability door's capture (`.wiki/alto/attn-score.md` §4; §1 is the
//! C++ lineage this reproduces).
//!
//! **A NEW FILE ON PURPOSE**, and the design says so in as many words: "the
//! accumulating capture-arm kernel is agent-built in a NEW FILE outside
//! `attn.rs`/`attn/kv.rs`" (§5). The ownership argument is the same one
//! `attn::dense` made: this entry shares the PAGES with the fa2 family and
//! nothing else — no schedule to walk, no split-kv partials, no merge, no
//! log-sum-exp plane, no `o`. It borrows the plan for exactly one thing, the
//! shape triple, and the plan's own `accepts` is the only question it asks
//! of it.
//!
//! **Where the module hangs, and why it is not `attn::score`.** As with
//! `attn::dense`: the FILE is `src/attn/score.rs`, where it belongs, but a
//! child module can only be declared by its parent and `src/attn.rs` is
//! closed to this wave. `lib.rs` re-homes the declaration with `#[path]`, so
//! the module path is `kernels_cuda::attn_score` until that seam reopens and
//! `pub mod score;` can move into `attn.rs`. Nothing about the op changes
//! when it does.
//!
//! One entry, one instantiation ladder, no workspace: the kernel recomputes
//! the softmax weights straight out of the pages rather than materialising
//! the `heads x window x kv_len` F32 slab the C++ lineage allocated per fire
//! (and refused above 1 GiB). See the unit's header for that argument.

use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/score.cuh";

/// Warps per block. Keys are split across them and folded once per window
/// row, so this is the kernel's only parallelism knob above the head — and
/// it is `attn/attention.cuh`'s post-processing width (256 threads), which
/// is the shape this capture reproduces in one pass.
const WARPS: u32 = 8;

const BLOCK: u32 = WARPS * 32;

/// The dot-product stamps, tightest first — `attn/dense.cuh`'s ladder, for
/// the same reason. A stamp is the unrolled per-lane length
/// (`stamp / 32` elements) and not a shape: the live head width may be
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

/// **PER-KEY ATTENTION MASS FOR AN OBSERVATION WINDOW, INTO A CALLER-OWNED
/// F32 SLAB.**
///
/// `q` is the capture window's query rows (`[rows, num_q_heads * head_dim]`,
/// bf16) paired with the window-REBASED `qo_indptr` — `i32`,
/// `[requests + 1]`, indexing `q.data` itself. `plan` is the very plan
/// [`prefill_lse`](crate::attn::prefill_lse) runs on and is read for one
/// thing only, the shape triple it was carved at; `pool` is the paged cache
/// this layer read.
///
/// For request `r` and query head `h` the output row is
/// `scores.ptr + ((lane_offset + r) * plane_stride + plane + h) * kv_max`
/// floats, and it holds
///
/// ```text
///   out[j] = (1 / rows) * sum over w of softmax_j( sm_scale * <q_w, k_j> )
/// ```
///
/// where `rows = min(observe, qo_len)`, `w` walks the request's LAST `rows`
/// query rows, and each row's softmax is taken over its own causal limit
/// `min(kv_len - rows + w + 1, kv_len)` — the arithmetic
/// `attn/attention.cuh`'s `attn_prefill_score_normalize` spells, taken here
/// in one pass. The result is a probability distribution over `[0, kv_len)`
/// summing to one: TOVA's number at `observe = 1`, SnapKV's at
/// `observe = 32`. The papers' extra fold over heads is deliberately not
/// taken — §4 rules the contract per-head and lets the guest fold.
///
/// **THE WHOLE ROW IS WRITTEN, ALWAYS.** `[kv_len, kv_max)` lands exactly
/// `0.0`, on every path including the degenerate ones (a request with no
/// pages, an empty cache, an empty window). The slab is reused across fires
/// and a stale tail is not "unset" — it is the previous fire's mass on keys
/// that no longer exist, which an eviction policy would rank on and never
/// fault.
///
/// # Errors
///
/// A refusal when:
///
/// - the plan was not carved at this head width, kv head count or window
///   ([`PrefillPlan::accepts`], asked first as `prefill_lse` asks it);
/// - **a sliding window is stated at all** — a windowed row is not the
///   softmax the eviction and interpretability papers define, and the
///   registry has refused capture under it since before this kernel existed
///   (`.wiki/alto/attn-score.md` §2.4 / §5). The refusal is semantic, not a
///   missing instantiation;
/// - the pool's key pages are not bf16 storage: this capture reads keys
///   directly out of the pages and dequantizes nothing;
/// - the slab is not F32, or its row is not `kv_max` wide;
/// - the query row width does not divide by the stated head width;
/// - the query heads do not group over the kv heads;
/// - the head is wider than the widest stamp;
/// - `observe` is zero — an observation window that observes nothing;
/// - `kv_max` is zero, or any stated extent overflows the kernel's `int`.
///
/// [`Error::DtypeUnsupported`] for a query in anything but bf16.
///
/// `kv_len > kv_max` is a caller error the engine refuses upstream and it is
/// NOT knowable here — `kv_len` is a device-side number read from the page
/// tables. The kernel is safe under it on its own: the softmax is still
/// taken over the true extent and only the store is clamped to `kv_max`.
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

    // The named refusals above judge the geometry a caller can state; these
    // are the landing contract, checked only once the fire is admissible.
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

    // The q row, then the block's two fold words per warp — the whole
    // workspace, and it is shared memory: the fire path allocates nothing
    // (design Article 7), which is the entire difference from the C++
    // lineage's per-fire `cudaMallocAsync`.
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
