//! `AttnScore`: per-key attention mass for an observation window — the alto
//! observability door's capture (`.wiki/alto/attn-score.md` §4; §1 is the C++
//! lineage this reproduces), and this plane's mirror of
//! `kernels_cuda::attn_score`.
//!
//! **A NEW FILE ON PURPOSE**, and the design says so in as many words: "the
//! accumulating capture-arm kernel is agent-built in a NEW FILE outside
//! `attn.rs`/`attn/kv.rs`" (§5). The ownership argument is [`dense`]'s: this
//! entry shares the PAGES with the sdpa family and nothing else — no
//! arbitration, no tile, no mask ladder, no log-sum-exp plane, no `o`. What it
//! borrows from the plan is one table, the position vector, and it borrows it
//! for the causal bound alone.
//!
//! Unlike the CUDA twin the module needs no `#[path]` re-homing: `attn.rs`
//! could take the one `pub mod score;` line, so the module path is the one the
//! design spells (`attn::score`) and no re-homing block is owed.
//!
//! One entry, one stamp ladder, no workspace: the kernel recomputes the
//! softmax weights straight out of the pages rather than materialising the
//! `heads x window x kv_len` F32 slab the C++ lineage allocated per fire (and
//! refused above 1 GiB). See the shader's header for that argument, and for
//! the one place this mirror's arithmetic is spelled differently from its
//! twin's without meaning anything different.
//!
//! **UNVERIFIED ON DEVICE.** The shader was written against its neighbours
//! (`attn/dense.metal` for the simdgroup fold, `attn/sdpa_paged.metal` for the
//! paged addressing) and against `kernels-cuda`'s `attn/score.cuh` for the
//! arithmetic. What the tests below pin is the host half — the ladder, the
//! refusals, the grid, the argument order.
//!
//! [`dense`]: crate::attn::dense

use dtype::Dtype;

use crate::attn::{KvPool, PrefillPlan, RaggedTensor};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/score.metal";

/// Simdgroups per threadgroup. Keys are split across them and folded once per
/// window row, so this is the kernel's only parallelism knob above the head —
/// and it is the CUDA twin's `WARPS`, which is the width this capture
/// reproduces.
const SIMDS: u32 = 8;

/// Threads per threadgroup — one Apple simdgroup is 32 lanes wide.
const THREADS: u32 = SIMDS * 32;

/// The dot-product stamps, tightest first — [`dense`](crate::attn::dense)'s
/// ladder and the twin's, for the same reason. A stamp is the unrolled
/// per-lane length (`stamp / 32` elements) and the threadgroup plane, not a
/// shape: the live head width may be anything at or below it, which is what
/// lets 64, 72 and 80 share the 128-wide stamp without any of them being
/// padded.
const STAMPS: [u32; 3] = [64, 128, 256];

/// The shipped point per stamp, in [`STAMPS`] order.
const CAPTURE: [&str; 3] = [
    "attn_score_capture_bfloat16_d_64",
    "attn_score_capture_bfloat16_d_128",
    "attn_score_capture_bfloat16_d_256",
];

/// The tightest stamp that holds this head, as an index into [`STAMPS`] — or
/// nothing, because a head wider than the last stamp is refused rather than
/// silently truncated.
fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

/// **PER-KEY ATTENTION MASS FOR AN OBSERVATION WINDOW, INTO A CALLER-OWNED
/// F32 SLAB.**
///
/// `q` is the capture window's query rows (`[rows, num_q_heads * head_dim]`,
/// bf16) paired with the window-REBASED `qo_indptr` — `i32`,
/// `[requests + 1]`, indexing `q.data` itself. `plan` is the very plan
/// [`prefill_lse`](crate::attn::arbiter::prefill_lse) runs on and is read for
/// one thing only, its position table, which is this plane's causal bound (the
/// shader's header says why that is the same number the twin reconstructs from
/// `kv_last_page_lens`). `pool` is the paged cache this layer read.
///
/// For request `r` and query head `h` the output row is
/// `scores` row `(lane_offset + r) * plane_stride + plane + h`, and it holds
///
/// ```text
///   out[j] = (1 / rows) * sum over w of softmax_j( sm_scale * <q_w, k_j> )
/// ```
///
/// where `rows = min(observe, qo_len)` and `w` walks the request's LAST `rows`
/// query rows, each row's softmax taken over its own causal limit. The result
/// is a probability distribution over the request's live KV summing to one:
/// TOVA's number at `observe = 1`, SnapKV's at `observe = 32`. The papers'
/// extra fold over heads is deliberately not taken — §4 rules the contract
/// per-head and lets the guest fold.
///
/// **THE WHOLE ROW IS WRITTEN, ALWAYS.** The tail past the live keys lands
/// exactly `0.0`, on every path including the degenerate ones (a request with
/// no pages, an empty cache, an empty window). The slab is reused across fires
/// and a stale tail is not "unset" — it is the previous fire's mass on keys
/// that no longer exist, which an eviction policy would rank on and never
/// fault.
///
/// # Errors
///
/// A refusal when:
///
/// - **a sliding window is stated at all** — a windowed row is not the softmax
///   the eviction and interpretability papers define, and the registry has
///   refused capture under it since before this kernel existed
///   (`.wiki/alto/attn-score.md` §2.4 / §5). The refusal is SEMANTIC, not a
///   missing instantiation, and it says so;
/// - **the pool's key pages are not bf16 storage** — this capture reads keys
///   directly out of the pages and dequantizes nothing, so a quantized or fp8
///   pool has no scores to give. Also semantic: there is no point to add;
/// - the stated head width is not the pool row's head stride, or the stated kv
///   head count is not the one the pool's strides spell
///   ([`kv_heads_agree`](crate::attn));
/// - the query row width does not divide by the stated head width, or the
///   query heads do not group over the kv heads;
/// - the head is wider than the widest stamp;
/// - the slab is not F32, or its row is not `kv_max` wide;
/// - the plan's position table is not `i32`, or the boundary vector is not;
/// - `observe` is zero — an observation window that observes nothing;
/// - `kv_max`, `page_size` or the request count is zero, or any stated extent
///   overflows the shader's `int`.
///
/// [`Error::DtypeUnsupported`] for a query in anything but bf16.
///
/// A live kv extent past `kv_max` is a caller error the engine refuses
/// upstream and it is NOT knowable here — the extent is a device-side number
/// read from the position and page tables. The kernel is safe under it on its
/// own: the softmax is still taken over the true extent and only the store is
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
    // The fire tables and the boundary vector are seats no op names, so the
    // trace-time validator never sees them and a disagreement is refused here
    // rather than asserted (the boundary rule at `crate::encode::refuse`).
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

    // The named refusals above judge the geometry a caller can state; these
    // are the landing contract, checked only once the fire is admissible.
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
    use crate::encode::ArgValue;
    use crate::probe::Probe;

    /// The ceiling the engine states today (`eta_ir::registry::
    /// ATTN_SCORE_KV_MAX`), spelled as a number because this crate names no
    /// registry: the entry takes the ceiling as an argument and has no opinion
    /// about which one, which is the whole reason it is a parameter.
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

    /// qwen35's attention shape: 16 query heads of 64 over 8 kv heads, two
    /// requests, the whole slab handed over as one rectangle.
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
        // Past the last stamp is not a wider point, it is no point.
        assert_eq!(stamp_for(257), None);
        assert_eq!(STAMPS.len(), CAPTURE.len());
    }

    /// The grid, the entry and every stated extent, in the order the shader
    /// declares them — the one thing a host test can pin about a kernel it
    /// cannot run.
    #[test]
    fn a_capturing_window_launches_one_group_per_request_per_head() {
        let probe = Probe::default();
        fire(&probe, None, Dtype::Bf16, slab(4 * 96)).expect("the capture enqueues");
        let (fired, args) = probe.only();
        assert_eq!(fired.file, FILE);
        assert_eq!(fired.entrypoint, "attn_score_capture_bfloat16_d_64");
        assert_eq!(fired.lanes, [2 * THREADS, 16, 1]);
        assert_eq!(fired.group, [THREADS, 1, 1]);
        assert_eq!(args[7], ArgValue::I32(16)); // page_size
        assert_eq!(args[8], ArgValue::I32(16)); // num_q_heads
        assert_eq!(args[9], ArgValue::I32(8)); // num_kv_heads
        assert_eq!(args[10], ArgValue::I32(64)); // head_dim
        assert_eq!(args[11], ArgValue::F32(0.125));
        assert_eq!(args[12], ArgValue::I32(32)); // observe
        assert_eq!(args[13], ArgValue::I32(0)); // lane_offset
        assert_eq!(args[14], ArgValue::I32(96)); // plane_stride
        assert_eq!(args[15], ArgValue::I32(0)); // plane
        assert_eq!(args[16], ArgValue::I32(KV_MAX as i32));
        // The slab is the only operand this entry writes.
        assert!(matches!(args[6], ArgValue::BufferMut(_)), "{:?}", args[6]);
    }

    /// **THE SLIDING-WINDOW REFUSAL IS SEMANTIC AND STAYS BY NAME** (design
    /// §2.4). It is not a missing instantiation, so it must not read like one.
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
                data: bf16(40, 2 * 512),
                indptr: i32s(3),
            },
            &plan(40),
            &pool(2, 512, Dtype::Bf16),
            None,
            512,
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
        .expect_err("a 512-wide head is past the ladder");
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
