//! `Attention`: paged sdpa over the fire's kv pool — the vector (decode) and
//! tiled (prefill) shaders, the appenders, and the plan payloads this plane's
//! attention carries. One entry per IR variant.
//!
//! The plans hold what the old plane's `AttnFireView` carried beside the
//! pool: the per-token fire tables (positions, owning request, mask) every
//! sdpa launch reads. Metal splits nothing — no partials, no split-k policy —
//! so a plan is tables, not workspaces, and building one encodes no device
//! work. The driver stores it behind `Box<dyn Any>` (design §6) and hands it
//! back to every fire built from it. The appenders are plan-free: they
//! address by the `write_page`/`write_offset` tables their ops state.
//!
//! The attention families that shared the old file keep their seats here:
//! [`mla`], [`index`], and [`pool`] are ported whole now, and the only typed
//! refusals left in the three are the two that name a SLAB THE LOAD DID NOT
//! RESERVE — which no plan naming the op can reach, because the reservation is
//! made from the op. [`ssm`], the recurrent mixer, sits beside them. `gate`
//! left for [`elemwise::gate`](crate::elemwise::gate):
//! it is elementwise, not attention.

pub mod arbiter;

/// The vision towers' bidirectional attention over the patch window — a
/// file of its own beside this one, sharing nothing with the paged family
/// but the word (`.wiki/alto/multimodal.md` §2). The CUDA twin had to be
/// re-homed by `#[path]` because its wave could not touch `attn.rs`; this
/// line is the declaration that one is owed.
pub mod dense;

pub mod merge;

/// qwen4's PLE n-gram hasher — a file of its own beside [`ssm`], whose
/// chunked shape it borrows: it is an `Attention` only by the clause that
/// counts a sequence cache, and everything else about it is per-token integer
/// arithmetic over token ids.
pub mod ple;

/// The alto observability door's capture — per-key attention mass over an
/// observation window, a file of its own beside this one for the reason
/// `.wiki/alto/attn-score.md` §5 states in as many words ("agent-built in a
/// NEW FILE outside `attn.rs`/`attn/kv.rs`").
pub mod score;

pub mod ssm;

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, head_grid, head_group, nonzero, refuse,
    stated,
};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/sdpa_paged.metal";

const SDPA_THREADS: u32 = 1024;

const SDPA_TILE: u32 = 32;

const SDPA_WIDTHS: [u32; 4] = [64, 128, 256, 512];

const SDPA_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

const SDPA_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

const SDPA_LSE_WIDTHS: [u32; 4] = [64, 128, 256, 512];

const SDPA_DECODE_LSE: [&str; 4] = [
    "sdpa_paged_decode_lse_bfloat16_d_64",
    "sdpa_paged_decode_lse_bfloat16_d_128",
    "sdpa_paged_decode_lse_bfloat16_d_256",
    "sdpa_paged_decode_lse_bfloat16_d_512",
];

const SDPA_TILED_LSE: [&str; 4] = [
    "sdpa_paged_tiled_lse_bfloat16_d_64",
    "sdpa_paged_tiled_lse_bfloat16_d_128",
    "sdpa_paged_tiled_lse_bfloat16_d_256",
    "sdpa_paged_tiled_lse_bfloat16_d_512",
];

/// What a decode fire needs beside the pool: the fire tables the vector sdpa
/// shader reads per token. Built once per fire by [`plan_decode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    /// `i32`, one per token: absolute position — the causal bound.
    pub positions: Tensor,

    /// `i32`, one per token: the owning request.
    pub request_of_token: Tensor,

    /// `u8` packed mask planes, one row per request.
    pub mask: Tensor,

    /// `u8`, one per request: whether its mask row is live.
    pub mask_enabled: Tensor,

    /// Elements from one request's mask row to the next.
    pub mask_stride: u32,
}

/// The prefill twin of [`DecodePlan`] — the tiled shader reads the same
/// tables, so the payloads agree; the types stay distinct because the IR
/// declares distinct struct kinds and the driver downcasts by them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    /// `i32`, one per token: absolute position — the causal bound.
    pub positions: Tensor,

    /// `i32`, one per token: the owning request.
    pub request_of_token: Tensor,

    /// `u8` packed mask planes, one row per request.
    pub mask: Tensor,

    /// `u8`, one per request: whether its mask row is live.
    pub mask_enabled: Tensor,

    /// Elements from one request's mask row to the next.
    pub mask_stride: u32,
}

/// The fire tables the driver binds beside the ops' named operands. They
/// reach the plan builders without an op naming them (`attention.masked`
/// names the mask alone, and only for its own launch), so the trace-time
/// validator never sees this binding — disagreement is refused, not
/// asserted (the boundary rule at [`refuse`]).
fn tables_agree(
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
) -> Result<(), Error> {
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's position table is {:?}, and this plan carries i32 positions",
                positions.dtype
            ),
        ));
    }
    if request_of_token.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's owning-request table is {:?}, and this plan carries an i32 \
                 request per token",
                request_of_token.dtype
            ),
        ));
    }
    if mask.dtype != Dtype::U8 {
        return Err(refuse(
            op,
            format!(
                "the fire's mask planes are {:?}, and this plan carries a packed u8 mask",
                mask.dtype
            ),
        ));
    }
    if mask_enabled.dtype != Dtype::U8 {
        return Err(refuse(
            op,
            format!(
                "the fire's mask-enabled flags are {:?}, and this plan carries one u8 \
                 per request",
                mask_enabled.dtype
            ),
        ));
    }
    if positions.rows != request_of_token.rows {
        return Err(refuse(
            op,
            format!(
                "the fire tables disagree: {} positions beside {} owning requests, and \
                 both are one entry per token",
                positions.rows, request_of_token.rows
            ),
        ));
    }
    Ok(())
}

/// Builds the decode plan. Metal derives no split policy and sizes no
/// partials — everything else the old plane computed rode on the operands —
/// so the context encodes nothing, and the build fails only by refusing
/// fire tables the driver bound wrong.
///
// MENLO-SEAM: `attention.plan_decode` in the IR names kv geometry —
// kv_indptr/kv_indices/last_page_len (never even seated here) and the
// `kv_len` this entry takes to stay aligned with the op — that this plan
// never reads: the pool row carries the page walk and the shaders bound by
// `positions`. The fire tables the plan does carry are not named by the op;
// the driver binds them from its fire state.
pub fn plan_decode(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<DecodePlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_decode",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(DecodePlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

/// Builds the prefill plan; see [`plan_decode`] — the same tables, the same
/// absence of device work.
///
// MENLO-SEAM: same misalignment as `plan_decode` — the op names kv geometry
// (`kv_len` seated, the rest never even taken) the plan never reads, and
// the fire tables come from driver fire state.
pub fn plan_prefill(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<PrefillPlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_prefill",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(PrefillPlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

fn head_point(op: &'static str, head_dim: u32, points: &[u32]) -> Result<usize, Error> {
    points
        .iter()
        .position(|&p| p == head_dim)
        .ok_or_else(|| refuse(op, format!("no sdpa shader is stamped at head width {head_dim}")))
}

/// The sliding extent the shader reads: 0 is "no window", so a stated window
/// of zero is a degenerate statement, not an unwindowed one.
fn window_extent(op: &'static str, window: Option<u32>) -> Result<i32, Error> {
    match window {
        None => Ok(0),
        Some(w) => {
            nonzero(op, "the sliding extent this attention states", w)?;
            stated(op, w)
        }
    }
}

/// The kv head count the pool row's strides spell, against the stated head
/// width. Pool strides are driver facts the validator never sees, so
/// disagreement is refused, not asserted.
fn pool_heads(op: &'static str, pool: &KvPool, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if pool.head_stride != u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the stated head width {head_dim} is not the pool row's head stride {}",
                pool.head_stride
            ),
        ));
    }
    if pool.seq_stride == 0 || !pool.seq_stride.is_multiple_of(pool.head_stride) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not a whole number of {head_dim}-wide kv heads",
                pool.seq_stride
            ),
        ));
    }
    u32::try_from(pool.seq_stride / pool.head_stride).map_err(|_| {
        refuse(
            op,
            "the kv head count this pool row's strides spell does not fit the shader's int",
        )
    })
}

fn kv_heads_agree(
    op: &'static str,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
) -> Result<(), Error> {
    let spelled = pool_heads(op, pool, head_dim)?;
    if kv_heads != spelled {
        return Err(refuse(
            op,
            format!(
                "the stated kv head count {kv_heads} is not the {spelled} the pool row's \
                 strides spell"
            ),
        ));
    }
    Ok(())
}

fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || width % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide query row does not divide by the stated head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

/// The paged shape one sdpa fire launches over, derived where the old plane
/// derived it: heads from the query row, kv heads from the pool strides.
struct Paged {
    q_heads: u32,

    kv_heads: u32,

    gqa: u32,

    window: i32,

    rows: u32,

    at: usize,
}

impl Paged {
    fn of(
        op: &'static str,
        q: Tensor,
        pool: &KvPool,
        window: Option<u32>,
        head_dim: u32,
    ) -> Result<Self, Error> {
        if pool.page_size <= 0 {
            return Err(refuse(op, "the kv page size is zero"));
        }
        let kv_heads = pool_heads(op, pool, head_dim)?;
        let q_heads = row_heads(op, q.width, head_dim)?;
        if q_heads % kv_heads != 0 {
            return Err(refuse(
                op,
                format!(
                    "the {q_heads} query heads this row divides into are not a whole number \
                     of the pool row's {kv_heads} kv heads"
                ),
            ));
        }
        Ok(Self {
            q_heads,
            kv_heads,
            gqa: q_heads / kv_heads,
            window: window_extent(op, window)?,
            rows: nonzero(op, "rows", q.rows)?,
            at: head_point(op, head_dim, &SDPA_WIDTHS)?,
        })
    }
}

fn lse_plane(op: &'static str, lse: Tensor, shape: &Paged) {
    debug_assert_eq!(lse.dtype, Dtype::F32, "`{op}` lands an f32 log-sum-exp plane");
    debug_assert!(
        lse.rows == shape.rows && lse.width == shape.q_heads,
        "`{op}`'s log-sum-exp plane is one f32 per head per row"
    );
}

fn vector_grid(op: &'static str, q_heads: u32, rows: u32) -> Result<[u32; 3], Error> {
    let x = q_heads.checked_mul(SDPA_THREADS).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {q_heads} query heads, one {SDPA_THREADS}-thread group each"),
        )
    })?;
    Ok([x, rows, 1])
}

fn tiled_grid(op: &'static str, q_heads: u32, rows: u32) -> Result<[u32; 3], Error> {
    let x = q_heads.checked_mul(SDPA_THREADS).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {q_heads} query heads, one {SDPA_THREADS}-thread group each"),
        )
    })?;
    Ok([x, rows.div_ceil(SDPA_TILE), 1])
}

/// One token-row per grid row: the decode shader.
#[allow(clippy::too_many_arguments)]
fn vector(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "the attention lands one output row per query row"
    );
    let shape = Paged::of(op, q, pool, window, head_dim)?;
    let entry = match lse {
        None => SDPA_DECODE[shape.at],
        Some(_) => SDPA_DECODE_LSE[head_point(op, head_dim, &SDPA_LSE_WIDTHS)?],
    };
    let mut args = vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        o.arg_mut(),
        stated(op, shape.gqa)?.arg(),
        plan.positions.arg(),
        plan.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        plan.mask.arg(),
        plan.mask_stride.arg(),
        plan.mask_enabled.arg(),
        shape.window.arg(),
        ctx.absent()?, // the sink seat; `attention.sink` folds that mass in afterwards
    ];
    if let Some(lse) = lse {
        lse_plane(op, lse, &shape);
        args.push(lse.arg_mut());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            vector_grid(op, shape.q_heads, shape.rows)?,
            [SDPA_THREADS, 1, 1],
        )),
        &args,
    )
}

/// [`SDPA_TILE`] token-rows per grid row: the prefill shader. `mask` is the
/// plane riding the shader's mask seat — the plan's own for the causal
/// entries (`mask_enabled` gates it per request), the op-named one for
/// [`masked`].
#[allow(clippy::too_many_arguments)]
fn tiled(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.dtype, { Bf16 => () });
    debug_assert!(
        o.rows == q.rows && o.width == q.width && o.dtype == q.dtype,
        "the attention lands one output row per query row"
    );
    let shape = Paged::of(op, q, pool, window, head_dim)?;
    let entry = match lse {
        None => SDPA_TILED[shape.at],
        Some(_) => SDPA_TILED_LSE[head_point(op, head_dim, &SDPA_LSE_WIDTHS)?],
    };
    let mut args = vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        o.arg_mut(),
        stated(op, shape.gqa)?.arg(),
        plan.positions.arg(),
        plan.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        mask.arg(),
        plan.mask_stride.arg(),
        plan.mask_enabled.arg(),
        shape.window.arg(),
        ctx.absent()?, // the sink seat; `attention.sink` folds that mass in afterwards
        stated(op, shape.rows)?.arg(),
    ];
    if let Some(lse) = lse {
        lse_plane(op, lse, &shape);
        args.push(lse.arg_mut());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            tiled_grid(op, shape.q_heads, shape.rows)?,
            [SDPA_THREADS, 1, 1],
        )),
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    vector(
        ctx,
        "attention.decode",
        q,
        pool,
        plan,
        window,
        head_dim,
        sm_scale,
        o,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    vector(
        ctx,
        "attention.decode_lse",
        q,
        pool,
        plan,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
}

/// The boundaries ride in `q`, but this shader walks the plan's
/// `request_of_token` instead — the indptr goes unread, as the old plane's
/// did.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    tiled(
        ctx, OP, q.data, pool, plan, plan.mask, window, head_dim, sm_scale, o, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    tiled(
        ctx,
        OP,
        q.data,
        pool,
        plan,
        plan.mask,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
}

/// Prefill against the op-named `mask` instead of the causal bound alone —
/// the same tiled shader, with `mask` in the seat the causal entries fill
/// from the plan.
///
// MENLO-SEAM: the plan carries mask tables of its own — every sdpa launch
// reads the mask seats, so the causal entries need them too — and the
// driver resolves `RuntimeInput::Mask` onto that same fire table: the
// op-named `mask` and `plan.mask` are one buffer wearing two names, and
// `mask_stride`/`mask_enabled` stay plan-carried because no op names them.
#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    if mask.dtype != Dtype::U8 {
        return Err(refuse(
            OP,
            format!(
                "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                mask.dtype
            ),
        ));
    }
    tiled(
        ctx, OP, q.data, pool, plan, mask, window, head_dim, sm_scale, o, None,
    )
}

/// Folds attention-sink mass into `o` using its log-sum-exp, in place on `o`.
pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "attn_sink_rescale_bfloat16" });
    debug_assert_eq!(lse.dtype, Dtype::F32, "`{OP}` reads an f32 log-sum-exp plane");
    let heads = row_heads(OP, o.width, head_dim)?;
    debug_assert!(
        lse.rows == o.rows && lse.width == heads,
        "`{OP}`'s log-sum-exp plane is one f32 per head per row"
    );
    let lanes = head_grid(OP, head_dim, heads, o.rows)?;
    ctx.fire(
        Fire::at("attn/attn_sink.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
        &[o.arg(), o.arg_mut(), lse.arg(), sink.arg()],
    )
}

/// Merges two attention readings over disjoint key sets by their
/// log-sum-exps — the fold lives in [`merge`].
#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    merge::merge_lse(ctx, o1, lse1, o2, lse2, heads, head_dim, o, lse)
}

/// `x = cap * tanh(x / cap)`, in place on `x`.
pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attention.logit_softcap";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "logit_softcap_bfloat16" });
    ctx.fire(
        Fire::at("attn/logit_softcap.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), x.arg_mut(), cap.arg()],
    )
}

/// The row split the pool strides spell for an appended `[heads x head_dim]`
/// row; strides are driver facts, so disagreement is refused.
fn head_split(op: &'static str, pool: &KvPool, row: u32) -> Result<(u32, u32), Error> {
    let head_dim = u32::try_from(pool.head_stride)
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            refuse(
                op,
                format!("the pool row's head stride {} spells no head width", pool.head_stride),
            )
        })?;
    if row == 0 || row % head_dim != 0 {
        return Err(refuse(
            op,
            format!("the {row}-wide appended row does not divide by the pool's head stride {head_dim}"),
        ));
    }
    let heads = row / head_dim;
    if pool.seq_stride != u64::from(heads) * u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not the page row this appender writes",
                pool.seq_stride
            ),
        ));
    }
    Ok((head_dim, heads))
}

#[allow(clippy::too_many_arguments)]
fn append_paged(
    ctx: &Ctx<'_>,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, k.dtype, { Bf16 => "kv_append_paged_bfloat16" });
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    debug_assert!(
        v.rows == k.rows && v.width == k.width && v.dtype == k.dtype,
        "the value plane is appended beside the key plane, one rectangle"
    );
    debug_assert!(
        write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
        "the write tables are u32: one destination page and one in-page slot per lane"
    );
    let (head_dim, heads) = head_split(op, pool, k.width)?;
    let lanes = head_grid(op, head_dim, heads, k.rows)?;
    ctx.fire(
        Fire::at("attn/kv_write.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            k.arg(),
            v.arg(),
            pool.keys.arg_mut(),
            pool.values.arg_mut(),
            ctx.absent()?, // the linear appender's position stream (buffer 4)
            stated(op, head_dim)?.arg(),
            ctx.absent()?, // …and its stride seats (buffers 6-9)
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            pool.page_size.arg(),
            ctx.absent()?, // buffer 11, unseated in the paged variant
            stated(op, heads)?.arg(),
            write_page.arg(),
            write_offset.arg(),
            0_i32.arg(), // src_row_stride: the appended rows are dense
        ],
    )
}

/// Appends `k`/`v` into the pool's pages, at the op-named write tables. The
/// write-geometry seam is closed: `attention.kv_append` states
/// `write_page`/`write_offset` itself, the appender reads exactly what the
/// op names, and the pool stays storage.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append_paged(
        ctx,
        "attention.kv_append",
        k,
        v,
        pool,
        write_page,
        write_offset,
    )
}

/// Appends one plane shared as both k and v, at the op-named write tables.
pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append_paged(
        ctx,
        "attention.kv_append_shared",
        plane,
        plane,
        pool,
        write_page,
        write_offset,
    )
}

/// `Mla`: multi-head latent attention — the metal mirror of `kernels-cuda`'s
/// `attn/mla.rs`/`attn/mla.cuh`. The projection prepare pipeline (latent
/// split/norm/rope, the per-head q-absorb, the paged latent appender) and the
/// naive simd flash engine are written here; the shaders live in
/// `attn/mla.metal`. Both the dense readers and the sparse (selected) ones —
/// the NSA index set's consumers — fire off that engine. The output-absorb
/// GEMM stays a typed refusal and names its own gap below.
pub mod mla {
    use dtype::Dtype;

    use crate::error::Error;
    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/mla.metal";

    /// One threadgroup per prepared row for the rms-reducing kernels.
    const PREP_THREADS: u32 = 256;

    /// The simdgroup width the flash kernel folds each key's dot over.
    const SIMD: u32 = 32;

    /// The latent-strip ceilings the flash kernel's register arrays hold
    /// (`kMaxCkvPer`/`kMaxKpePer` in the shader): rank up to 512, rope up to
    /// 128, both a whole number of 32-lane strips.
    const MAX_CKV: u32 = 16 * SIMD;
    const MAX_KPE: u32 = 4 * SIMD;

    /// Carries no device state: the metal flash engine reads the fire's
    /// position/owning-request tables and the pool's page walk directly at each
    /// attention arm, the way the paged sdpa family does — so the plan an
    /// `attention.mla_plan` builds is empty, and its only role is the struct
    /// value decode and prefill name. (The CUDA plan sizes an fa2 workspace;
    /// the naive engine this plane runs needs none.)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MlaPlan;

    // MENLO-SEAM (kernel side): the op names kv geometry
    // (kv_indptr/kv_indices/last_page_len/kv_len) that this plane's attention
    // never reads — the pool row carries the page walk and the shaders bound by
    // `positions` — so the builder takes them only to stay shaped like the
    // paged plan builders and encodes nothing.
    pub fn plan(
        _ctx: &Ctx<'_>,
        _kv_indptr: Tensor,
        _kv_indices: Tensor,
        _last_page_len: Tensor,
        _kv_len: Tensor,
    ) -> Result<MlaPlan, Error> {
        Ok(MlaPlan)
    }

    /// Splits `kv_a` into the rmsnormed compressed latent (`kv_c`, the first
    /// `kv_lora_rank` lanes) and the rope tail (`k_pe`, the remainder). Mirrors
    /// `mla.cuh`'s `mla_latents`.
    #[allow(clippy::too_many_arguments)]
    pub fn latents(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        split_kv_a_norm(ctx, "attention.mla_latents", kv_a, weight, eps, kv_lora_rank, kv_c, k_pe)
    }

    /// [`latents`], then a partial rope over the rope tail — the same
    /// `rope::partial_q` the CUDA entry rotates `k_pe` with, at head width
    /// `rope_dim` (the tail is one rope-wide plane per row).
    #[allow(clippy::too_many_arguments)]
    pub fn latents_rope(
        ctx: &Ctx<'_>,
        kv_a: Tensor,
        positions: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        split_kv_a_norm(
            ctx,
            "attention.mla_latents_rope",
            kv_a,
            weight,
            eps,
            kv_lora_rank,
            kv_c,
            k_pe,
        )?;
        crate::elemwise::rope::partial_q(ctx, k_pe, positions, rope_dim, rope_dim, theta)
    }

    #[allow(clippy::too_many_arguments)]
    fn split_kv_a_norm(
        ctx: &Ctx<'_>,
        op: &'static str,
        kv_a: Tensor,
        weight: Tensor,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Tensor,
        k_pe: Tensor,
    ) -> Result<(), Error> {
        let entry = dtype_dispatch!(op, kv_a.dtype, { Bf16 => "mla_latents_bfloat16" });
        debug_assert!(
            kv_c.width == kv_lora_rank && kv_c.rows == kv_a.rows,
            "the latent output is the stated rank wide, one row per source row"
        );
        debug_assert!(
            k_pe.rows == kv_a.rows,
            "the rope tail is one row per source row"
        );
        let kv_lora = stated(op, kv_lora_rank)?;
        let rope = stated(op, k_pe.width)?;
        let src_row_stride = stated(op, kv_a.width)?;
        if kv_a.width < kv_lora_rank + k_pe.width {
            return Err(refuse(
                op,
                format!(
                    "the {}-wide source row does not hold the {kv_lora_rank}-wide latent beside \
                     the {}-wide rope tail",
                    kv_a.width, k_pe.width
                ),
            ));
        }
        let rows = crate::encode::nonzero(op, "rows", kv_a.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([PREP_THREADS * rows, 1, 1], [PREP_THREADS, 1, 1])),
            &[
                kv_a.arg(),
                weight.arg(),
                kv_c.arg_mut(),
                k_pe.arg_mut(),
                kv_lora.arg(),
                rope.arg(),
                src_row_stride.arg(),
                eps.arg(),
            ],
        )
    }

    /// Splits `q_b` into per-head nope (`q_nope`) and rope (`q_pe`) planes —
    /// one thread per source element, mirroring `mla.cuh`'s `mla_split_q_b`.
    #[allow(clippy::too_many_arguments)]
    pub fn split_q_b(
        ctx: &Ctx<'_>,
        q_b: Tensor,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Tensor,
        q_pe: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_split_q_b";
        let entry = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "mla_split_q_b_bfloat16" });
        let heads = stated(OP, heads)?;
        let nope = stated(OP, nope_dim)?;
        let rope = stated(OP, rope_dim)?;
        let per = i64::from(nope) + i64::from(rope);
        let total = i64::from(q_b.rows) * i64::from(heads) * per;
        let total = i32::try_from(total)
            .map_err(|_| refuse(OP, format!("{total} split elements do not fit the shader's int")))?;
        let lanes = u32::try_from(total)
            .map_err(|_| refuse(OP, "the split grid will not launch"))?;
        debug_assert!(
            q_nope.width == u32::try_from(i64::from(heads) * i64::from(nope)).unwrap_or(u32::MAX)
                && q_pe.width == u32::try_from(i64::from(heads) * i64::from(rope)).unwrap_or(u32::MAX),
            "the nope and rope planes are the per-head cut of q_b's row"
        );
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([lanes, 1, 1], [256, 1, 1])),
            &[
                q_b.arg(),
                q_nope.arg_mut(),
                q_pe.arg_mut(),
                total.arg(),
                heads.arg(),
                nope.arg(),
                rope.arg(),
            ],
        )
    }

    /// Absorbs `kv_b`'s up-projection into q: the per-head matmul
    /// `q_latent[t,h,:] = q_nope[t,h,:] · kv_b_nope[h]`, where `kv_b` is the
    /// `[heads·(nope+v_dim), rank]` checkpoint weight. Mirrors `mla.cuh`'s
    /// `absorb_q` batched GEMM as one thread per output element.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_q(
        ctx: &Ctx<'_>,
        q_nope: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_absorb_q";
        let entry = dtype_dispatch!(OP, q_nope.dtype, { Bf16 => "mla_absorb_q_bfloat16" });
        let heads_i = stated(OP, heads)?;
        let rank = stated(OP, kv_lora_rank)?;
        let nope = stated(OP, nope_dim)?;
        let v_dim = stated(OP, v_head_dim)?;
        let rows = crate::encode::nonzero(OP, "rows", q_nope.rows)?;
        debug_assert!(
            q_latent.width == heads * kv_lora_rank && q_latent.rows == q_nope.rows,
            "the absorbed q is `heads · rank` wide, one row per token"
        );
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([kv_lora_rank, heads, rows], [SIMD.min(kv_lora_rank), 1, 1])),
            &[
                q_nope.arg(),
                kv_b.arg(),
                q_latent.arg_mut(),
                heads_i.arg(),
                rank.arg(),
                nope.arg(),
                v_dim.arg(),
            ],
        )
    }

    /// The absorb's other half — the latent attention reading mapped back
    /// through `kv_b`'s value planes: `o[t,h,:] = latent[t,h,:] · W_UV[h]ᵀ`,
    /// where `W_UV[h]` is the `[v_dim, rank]` block sitting immediately after
    /// head `h`'s `[nope, rank]` key-up block inside the same
    /// `[heads·(nope+v_dim), rank]` checkpoint weight `absorb_q` reads.
    ///
    /// **THE V-BLOCK BASE, WHICH THIS ENTRY WAS ONCE DEFERRED OVER.**
    /// `mla.rs` starts its A operand at `kv_b.ptr.wrapping_add(2·nope·rank)`,
    /// which read as an element count is not the standard packing and is what
    /// stopped the port. `Tensor::ptr` is a device ADDRESS and that add is in
    /// BYTES (`kernels-cuda`'s own `plane_bytes = rows·width·2` guard is
    /// written in the same units); the `2` is `sizeof(bf16)`. The base is
    /// `nope·rank` ELEMENTS, the per-head stride is `(nope+v_dim)·rank`, and
    /// the packing is exactly the standard one — heads outer, the key-up and
    /// value-up blocks contiguous within a head, each row `rank` wide.
    /// `engine-metal/tests/mla_on_device.rs` measures it end to end against
    /// the unabsorbed attention rather than resting on the reading.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb_out(
        ctx: &Ctx<'_>,
        latent: Tensor,
        kv_b: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_absorb_out";
        let entry = dtype_dispatch!(OP, latent.dtype, { Bf16 => "mla_absorb_out_bfloat16" });
        let heads_i = stated(OP, heads)?;
        let rank = stated(OP, kv_lora_rank)?;
        // The value width sizes the threadgroup as well as the grid, so a zero
        // is refused by name rather than launched as an empty group.
        let v_dim = stated(OP, crate::encode::nonzero(OP, "the value head dim", v_head_dim)?)?;
        let nope = stated(OP, nope_dim)?;
        let rows = crate::encode::nonzero(OP, "rows", latent.rows)?;
        debug_assert!(
            latent.width == heads * kv_lora_rank,
            "the latent reading is `heads · rank` wide, one row per token"
        );
        debug_assert!(
            o.width == heads * v_head_dim && o.rows == latent.rows,
            "the value-space output is `heads · v_dim` wide, one row per token"
        );
        ctx.fire(
            Fire::at(FILE, entry)
                .apply(Grid::of([v_head_dim, heads, rows], [SIMD.min(v_head_dim), 1, 1])),
            &[
                latent.arg(),
                kv_b.arg(),
                o.arg_mut(),
                heads_i.arg(),
                rank.arg(),
                v_dim.arg(),
                nope.arg(),
            ],
        )
    }

    /// Appends one latent row (`kv_c` beside `k_pe`) into the pool's pages at
    /// the op-named write tables. `kv_c` lands in the keys pages (rank-wide,
    /// one kv head), `k_pe` in the values pages (rope-wide).
    pub fn kv_append(
        ctx: &Ctx<'_>,
        kv_c: Tensor,
        k_pe: Tensor,
        pool: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.mla_kv_append";
        let entry = dtype_dispatch!(OP, kv_c.dtype, { Bf16 => "mla_kv_append_bfloat16" });
        if pool.page_size <= 0 {
            return Err(refuse(OP, "the kv page size is zero"));
        }
        debug_assert!(
            k_pe.rows == kv_c.rows,
            "the rope plane is appended beside the latent plane, one row each"
        );
        debug_assert!(
            write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
            "the write tables are u32: one destination page and one in-page slot per row"
        );
        let rows = crate::encode::nonzero(OP, "rows", kv_c.rows)?;
        let width = kv_c.width.max(k_pe.width);
        let width = crate::encode::nonzero(OP, "the appended latent width", width)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([width, rows, 1], [width.min(256), 1, 1])),
            &[
                kv_c.arg(),
                k_pe.arg(),
                pool.keys.arg_mut(),
                pool.values.arg_mut(),
                write_page.arg(),
                write_offset.arg(),
                pool.page_size.arg(),
                stated(OP, kv_c.width)?.arg(),
                stated(OP, k_pe.width)?.arg(),
            ],
        )
    }

    /// The naive paged flash engine all four readers share: one simdgroup per
    /// (head, query row), an online-softmax sweep over `[0, position]` of the
    /// latent kv. `positions`/`request_of_token` are the fire tables the driver
    /// binds beside the pool — the causal bound and the owning request per row.
    ///
    /// `selection`, when present, is the `i32` index row
    /// `attention.index_topk` published for this token: the sweep then walks
    /// that row's `top_k` entries instead of `[0, position]`, dropping every
    /// entry outside the causal bound (the -1 padded tail included). This is
    /// `mla_naive_paged_kernel`'s nullable `selection`/`top_k` pair, which on
    /// this plane picks the shader entrypoint rather than a null pointer —
    /// metal cannot leave a declared buffer seat empty.
    #[allow(clippy::too_many_arguments)]
    fn flash(
        ctx: &Ctx<'_>,
        op: &'static str,
        q: Tensor,
        q_pe: Tensor,
        selection: Option<Tensor>,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        let entry = match selection {
            None => dtype_dispatch!(op, q.dtype, { Bf16 => "mla_naive_paged_bfloat16" }),
            Some(_) => {
                dtype_dispatch!(op, q.dtype, { Bf16 => "mla_naive_paged_selected_bfloat16" })
            }
        };
        if pool.page_size <= 0 {
            return Err(refuse(op, "the kv page size is zero"));
        }
        let heads = crate::encode::nonzero(op, "the head count this attention states", heads)?;
        let rows = crate::encode::nonzero(op, "rows", q.rows)?;
        let ckv = latent_strip(op, "latent rank", kv_lora_rank, MAX_CKV)?;
        if q_pe.width == 0 || q_pe.width % heads != 0 {
            return Err(refuse(
                op,
                format!("the {}-wide rotated q plane does not divide by the {heads} heads", q_pe.width),
            ));
        }
        let kpe = latent_strip(op, "rope width", q_pe.width / heads, MAX_KPE)?;
        debug_assert!(
            positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
            "the fire's position and owning-request tables are i32, one entry per row"
        );
        debug_assert!(
            o.rows == q.rows && o.width == heads * kv_lora_rank,
            "the latent reading is `heads · rank` wide, one row per query row"
        );
        let mut args = vec![
            q.arg(),
            q_pe.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            o.arg_mut(),
            positions.arg(),
            request_of_token.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.page_size.arg(),
            stated(op, heads)?.arg(),
            ckv.arg(),
            kpe.arg(),
            sm_scale.arg(),
        ];
        if let Some(selection) = selection {
            // `mla.rs`'s two selection guards, verbatim: an `i32` row per query
            // row, and the budget the shader strides it by is its width.
            debug_assert!(
                selection.dtype == Dtype::I32,
                "the selection is an i32 key-index row"
            );
            if selection.rows != o.rows {
                return Err(refuse(
                    op,
                    "the selection does not carry one row per query row",
                ));
            }
            args.push(selection.arg());
            args.push(stated(op, selection.width)?.arg());
        }
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([heads * SIMD, rows, 1], [SIMD, 1, 1])),
            &args,
        )
    }

    /// A latent width the flash kernel can lane-split: a nonzero whole number
    /// of 32-lane strips, at most `max` (the register-array ceiling).
    fn latent_strip(op: &'static str, what: &str, width: u32, max: u32) -> Result<i32, Error> {
        if width == 0 || width % SIMD != 0 || width > max {
            return Err(refuse(
                op,
                format!(
                    "the {what} {width} is not one this kernel can lane-split \
                     (a nonzero multiple of {SIMD}, at most {max})"
                ),
            ));
        }
        stated(op, width)
    }

    /// Latent attention over one token per row (decode). The plan carries no
    /// device state — the fire tables reach the kernel through
    /// `positions`/`request_of_token`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        ctx: &Ctx<'_>,
        q: Tensor,
        q_pe: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_decode",
            q,
            q_pe,
            None,
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    /// Latent attention over ragged prefixes (prefill); same flash engine, the
    /// causal bound per row read from `positions`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        ctx: &Ctx<'_>,
        q: RaggedTensor,
        q_pe: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_prefill",
            q.data,
            q_pe,
            None,
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    /// Decode over the sparse selection `attention.index_topk` produced: the
    /// same flash engine, its key sweep handed the index row instead of the
    /// causal range. `selection` is `[rows, top_k]` i32, ascending key ids with
    /// a -1 padded tail.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_selected(
        ctx: &Ctx<'_>,
        q: Tensor,
        q_pe: Tensor,
        selection: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_decode_selected",
            q,
            q_pe,
            Some(selection),
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    /// Prefill over the sparse selection; same engine, same selection row, the
    /// causal bound per row read from `positions` as the dense prefill reads it.
    ///
    /// **CAUSALITY IS ONE BOUND HERE, NOT TWO.** `mla.rs` splits the two
    /// selected entries by a `causal` flag (`false` for decode, `true` for
    /// prefill) that only ever chooses between `kv_len` and `abs_q + 1`. This
    /// plane's engine bounds by `positions[row] + 1` for both, which is
    /// `abs_q + 1` exactly, and for a decode row — whose position IS the last
    /// cached slot — is `kv_len` exactly. The divergence the file header states
    /// for the dense readers covers these two, unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_selected(
        ctx: &Ctx<'_>,
        q: RaggedTensor,
        q_pe: Tensor,
        selection: Tensor,
        pool: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Tensor,
    ) -> Result<(), Error> {
        flash(
            ctx,
            "attention.mla_prefill_selected",
            q.data,
            q_pe,
            Some(selection),
            pool,
            positions,
            request_of_token,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        )
    }

    /// The key walk a selected reader performs, in host arithmetic — the
    /// deviceless pin on the ONE part of `mla_naive_paged_selected` that is a
    /// semantic rather than a launch.
    ///
    /// **IT IS HERE SO THE SELECTION READ CAN BE TESTED WITHOUT A GPU**, the
    /// way [`super::index::bisect_select`] pins the selection *write*. Nothing
    /// on the fire path calls it: the sweep runs on the device, in the shader,
    /// over the row `index_topk_paged` wrote. Every line has a line above it in
    /// `attn/mla.metal`'s body and in `mla.cuh`'s `mla_naive_paged_kernel` —
    /// `top_k` steps, `j = srow[n]`, and `continue` on any `j` outside
    /// `[0, j_end)`, which is both the -1 padded tail and any id the causal
    /// bound does not reach.
    #[must_use]
    pub fn selected_sweep(selection: &[i32], j_end: i32) -> Vec<i32> {
        let mut keys = Vec::with_capacity(selection.len());
        for &j in selection {
            if j < 0 || j >= j_end {
                continue;
            }
            keys.push(j);
        }
        keys
    }

    /// The online-softmax accumulation the flash body runs over the keys a
    /// sweep names, in host arithmetic: `m`/`lsum` rescaled per key, the
    /// value strip accumulated in the same pass, normalized once at the end.
    ///
    /// Also deviceless-only, and here for one claim: that the streaming form
    /// the shader runs equals the batch softmax over the same key set, so a
    /// selected reading equals the dense reading restricted to the selected
    /// keys. `scores` are the pre-softmax logits (already `sm_scale`d) and
    /// `values` the matching value rows, `width` wide.
    #[must_use]
    pub fn flash_reading(scores: &[f32], values: &[f32], width: usize) -> Vec<f32> {
        let mut acc = vec![0.0f32; width];
        let mut m = -3.0e38f32;
        let mut lsum = 0.0f32;
        for (n, &score) in scores.iter().enumerate() {
            let m_new = m.max(score);
            let corr = (m - m_new).exp();
            let p = (score - m_new).exp();
            lsum = lsum * corr + p;
            for (i, a) in acc.iter_mut().enumerate() {
                *a = *a * corr + p * values[n * width + i];
            }
            m = m_new;
        }
        let inv = if lsum > 0.0 { 1.0 / lsum } else { 0.0 };
        for a in &mut acc {
            *a *= inv;
        }
        acc
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::encode::ArgValue;
        use crate::probe::Probe;

        const RANK: u32 = 512;
        const ROPE: u32 = 64;
        const NOPE: u32 = 128;
        const HEADS: u32 = 4;

        fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
            Tensor::new(buf, rows, width, Dtype::Bf16)
        }

        fn i32t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::I32)
        }

        fn u32t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::U32)
        }

        /// A latent pool: `keys` are the rank-wide ckv pages, `values` the
        /// rope-wide kpe pages — the split `mla.cuh`'s `Layer` reads.
        fn latent_pool() -> KvPool {
            KvPool {
                keys: bf16(30, 4096, RANK),
                values: bf16(31, 4096, ROPE),
                page_indices: u32t(32, 64),
                page_indptr: u32t(33, 8),
                page_size: 16,
                seq_stride: u64::from(RANK),
                head_stride: u64::from(RANK),
            }
        }

        /// **THE LATENT SPLIT IS ONE THREADGROUP PER ROW**, over `mla.metal`'s
        /// rms-reducing entry, with the rank/rope/stride/eps the shader reads.
        #[test]
        fn latents_marshals_a_per_row_reduce() {
            let probe = Probe::default();
            let kv_a = bf16(1, 3, RANK + ROPE);
            latents(&probe, kv_a, bf16(2, 1, RANK), 1e-6, RANK, bf16(3, 3, RANK), bf16(4, 3, ROPE))
                .expect("the latent split enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_latents_bfloat16");
            assert_eq!(f.file, "attn/mla.metal");
            assert_eq!(f.lanes, [PREP_THREADS * 3, 1, 1]);
            assert_eq!(f.group, [PREP_THREADS, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1));
            assert_eq!(a[1], ArgValue::Buffer(2));
            assert_eq!(a[2], ArgValue::BufferMut(3));
            assert_eq!(a[3], ArgValue::BufferMut(4));
            assert_eq!(a[4], ArgValue::I32(RANK as i32));
            assert_eq!(a[5], ArgValue::I32(ROPE as i32));
            assert_eq!(a[6], ArgValue::I32((RANK + ROPE) as i32));
            assert_eq!(a[7], ArgValue::F32(1e-6));
        }

        /// The `_rope` twin is the split THEN a proportional neox over the tail
        /// — two launches, `mla.cuh`'s `latents` + `rope::partial_q`.
        #[test]
        fn latents_rope_is_the_split_then_a_tail_rotation() {
            let probe = Probe::default();
            let kv_a = bf16(1, 2, RANK + ROPE);
            latents_rope(
                &probe, kv_a, i32t(9, 2), bf16(2, 1, RANK), 1e-6, RANK, ROPE, 10000.0,
                bf16(3, 2, RANK), bf16(4, 2, ROPE),
            )
            .expect("the roped split enqueues");
            let fires = probe.fires();
            assert_eq!(fires.len(), 2, "the split and the tail rotation");
            assert_eq!(fires[0].0.entrypoint, "mla_latents_bfloat16");
            assert_eq!(fires[1].0.entrypoint, "neox_prop_mb_bfloat16");
            assert_eq!(fires[1].0.file, "elemwise/rope_neox.metal");
            // The rotation turns the tail in place: k_pe is the mutated buffer.
            assert_eq!(fires[1].1[0], ArgValue::BufferMut(4));
        }

        /// The q split is one thread per source element; the flat lane count is
        /// `rows · heads · (nope + rope)`.
        #[test]
        fn split_q_b_is_flat_over_every_source_element() {
            let probe = Probe::default();
            let per = NOPE + ROPE;
            let q_b = bf16(1, 2, HEADS * per);
            split_q_b(&probe, q_b, HEADS, NOPE, ROPE, bf16(2, 2, HEADS * NOPE), bf16(3, 2, HEADS * ROPE))
                .expect("the q split enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_split_q_b_bfloat16");
            let total = 2 * HEADS * per;
            assert_eq!(f.lanes, [total, 1, 1]);
            assert_eq!(a[3], ArgValue::I32(total as i32));
            assert_eq!(a[4], ArgValue::I32(HEADS as i32));
            assert_eq!(a[5], ArgValue::I32(NOPE as i32));
            assert_eq!(a[6], ArgValue::I32(ROPE as i32));
        }

        /// The latent appender writes both planes by the op-named write tables:
        /// ckv into the keys pages, kpe into the values pages.
        #[test]
        fn kv_append_addresses_both_planes_by_the_write_tables() {
            let probe = Probe::default();
            let pool = latent_pool();
            kv_append(&probe, bf16(1, 3, RANK), bf16(2, 3, ROPE), &pool, u32t(5, 3), u32t(6, 3))
                .expect("the latent append enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_kv_append_bfloat16");
            assert_eq!(f.lanes, [RANK, 3, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1));
            assert_eq!(a[1], ArgValue::Buffer(2));
            assert_eq!(a[2], ArgValue::BufferMut(30)); // keys pages == ckv
            assert_eq!(a[3], ArgValue::BufferMut(31)); // values pages == kpe
            assert_eq!(a[4], ArgValue::Buffer(5));
            assert_eq!(a[5], ArgValue::Buffer(6));
            assert_eq!(a[6], ArgValue::I32(16));
            assert_eq!(a[7], ArgValue::I32(RANK as i32));
            assert_eq!(a[8], ArgValue::I32(ROPE as i32));
        }

        /// The q-absorb is one thread per `(rank lane, head, token)`, over the
        /// `[heads·(nope+v_dim), rank]` weight.
        #[test]
        fn absorb_q_is_one_thread_per_latent_output() {
            let probe = Probe::default();
            let q_nope = bf16(1, 2, HEADS * NOPE);
            absorb_q(&probe, q_nope, bf16(2, HEADS * (NOPE + NOPE), RANK), HEADS, RANK, NOPE, NOPE, bf16(3, 2, HEADS * RANK))
                .expect("the q absorb enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_absorb_q_bfloat16");
            assert_eq!(f.lanes, [RANK, HEADS, 2]);
            assert_eq!(f.group, [SIMD, 1, 1]);
            assert_eq!(a[3], ArgValue::I32(HEADS as i32));
            assert_eq!(a[4], ArgValue::I32(RANK as i32));
            assert_eq!(a[5], ArgValue::I32(NOPE as i32));
            assert_eq!(a[6], ArgValue::I32(NOPE as i32));
        }

        /// The dense reader is one simdgroup per `(head, row)`; keys/values ride
        /// the ckv/kpe seats, and the causal bound rides `positions`.
        #[test]
        fn decode_is_a_simdgroup_per_head_row() {
            let probe = Probe::default();
            let pool = latent_pool();
            attention_decode(
                &probe, bf16(1, 2, HEADS * RANK), bf16(2, 2, HEADS * ROPE), &pool,
                i32t(7, 2), i32t(8, 2), HEADS, RANK, 0.5, bf16(3, 2, HEADS * RANK),
            )
            .expect("the latent decode enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_naive_paged_bfloat16");
            assert_eq!(f.lanes, [HEADS * SIMD, 2, 1]);
            assert_eq!(f.group, [SIMD, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // q_latent
            assert_eq!(a[1], ArgValue::Buffer(2)); // q_pe
            assert_eq!(a[2], ArgValue::Buffer(30)); // ckv pages
            assert_eq!(a[3], ArgValue::Buffer(31)); // kpe pages
            assert_eq!(a[4], ArgValue::BufferMut(3)); // o
            assert_eq!(a[5], ArgValue::Buffer(7)); // positions
            assert_eq!(a[6], ArgValue::Buffer(8)); // request_of_token
            assert_eq!(a[9], ArgValue::I32(16)); // page_size
            assert_eq!(a[10], ArgValue::I32(HEADS as i32));
            assert_eq!(a[11], ArgValue::I32(RANK as i32));
            assert_eq!(a[12], ArgValue::I32(ROPE as i32));
            assert_eq!(a[13], ArgValue::F32(0.5));
        }

        /// Prefill is the same flash engine over `q.data`.
        #[test]
        fn prefill_shares_the_flash_engine() {
            let probe = Probe::default();
            let pool = latent_pool();
            let q = RaggedTensor {
                data: bf16(1, 5, HEADS * RANK),
                indptr: i32t(9, 3),
            };
            attention_prefill(
                &probe, q, bf16(2, 5, HEADS * ROPE), &pool, i32t(7, 5), i32t(8, 5),
                HEADS, RANK, 0.5, bf16(3, 5, HEADS * RANK),
            )
            .expect("the latent prefill enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_naive_paged_bfloat16");
            assert_eq!(f.lanes, [HEADS * SIMD, 5, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // q.data, not the indptr
        }

        /// A latent rank the register strips cannot hold is refused by name,
        /// not launched.
        #[test]
        fn a_rank_the_strips_cannot_hold_is_refused() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode(
                &probe, bf16(1, 1, HEADS * 500), bf16(2, 1, HEADS * ROPE), &pool,
                i32t(7, 1), i32t(8, 1), HEADS, 500, 0.5, bf16(3, 1, HEADS * 500),
            )
            .expect_err("500 is not a multiple of 32");
            assert!(format!("{why}").contains("lane-split"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// The output-absorb is one thread per `(value lane, head, token)`,
        /// over the same `[heads·(nope+v_dim), rank]` weight the q-absorb
        /// reads — and its trailing pair is `v_dim, nope`, the reverse of the
        /// q-absorb's `nope, v_dim`, because that is the order `mla.rs`'s two
        /// entries take them in and the seat order no type can catch.
        #[test]
        fn absorb_out_is_one_thread_per_value_output() {
            const VDIM: u32 = 128;
            let probe = Probe::default();
            let latent = bf16(1, 2, HEADS * RANK);
            absorb_out(&probe, latent, bf16(2, HEADS * (NOPE + VDIM), RANK), HEADS, RANK, VDIM, NOPE, bf16(3, 2, HEADS * VDIM))
                .expect("the output absorb enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_absorb_out_bfloat16");
            assert_eq!(f.file, "attn/mla.metal");
            assert_eq!(f.lanes, [VDIM, HEADS, 2]);
            assert_eq!(f.group, [SIMD, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1));
            assert_eq!(a[1], ArgValue::Buffer(2));
            assert_eq!(a[2], ArgValue::BufferMut(3));
            assert_eq!(a[3], ArgValue::I32(HEADS as i32));
            assert_eq!(a[4], ArgValue::I32(RANK as i32));
            assert_eq!(a[5], ArgValue::I32(VDIM as i32));
            assert_eq!(a[6], ArgValue::I32(NOPE as i32));
        }

        /// A zero value width sizes an empty threadgroup as well as an empty
        /// grid; it is refused by name rather than launched.
        #[test]
        fn a_zero_value_width_is_refused_rather_than_launched() {
            let probe = Probe::default();
            let why = absorb_out(&probe, bf16(1, 1, HEADS * RANK), bf16(2, HEADS * NOPE, RANK), HEADS, RANK, 0, NOPE, bf16(3, 1, 1))
                .expect_err("a zero value head dim");
            assert!(format!("{why}").contains("value head dim"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// **THE V-BLOCK BASE, IN HOST ARITHMETIC.** The one number the port
        /// was deferred over, stated where it can be read without a GPU: the
        /// CUDA entry's `wv = kv_b.ptr.wrapping_add(2·nope·rank)` is a BYTE
        /// add over a device address, so at `sizeof(bf16) == 2` the value-up
        /// block begins `nope·rank` ELEMENTS in — the standard packing, and
        /// the offset the shader spells. The device gate measures it; this
        /// records the reconciliation so the byte/element reading cannot be
        /// lost again.
        #[test]
        fn the_value_block_base_is_the_byte_add_read_as_elements() {
            let (nope, v_dim, rank) = (128u64, 128u64, 512u64);
            let cuda_byte_offset = 2 * nope * rank;
            let elements = cuda_byte_offset / 2;
            assert_eq!(elements, nope * rank, "the `2` is sizeof(bf16), not a stride");
            // Head h's value block, as the shader addresses it.
            for h in 0..4u64 {
                let shader = h * (nope + v_dim) * rank + nope * rank;
                let cuda = h * (nope + v_dim) * rank + elements;
                assert_eq!(shader, cuda, "head {h}'s value block is one place");
            }
        }

        /// A selection row, `TOPK` wide, one per query row.
        fn sel(buf: u32, rows: u32, top_k: u32) -> Tensor {
            Tensor::new(buf, rows, top_k, Dtype::I32)
        }

        /// **THE SELECTED READER IS THE DENSE ONE PLUS TWO SEATS.** Same
        /// engine, same grid, the first fourteen arguments unmoved; the index
        /// plane and its budget ride behind them, and the entrypoint is the
        /// sparse point.
        #[test]
        fn decode_selected_is_the_dense_launch_with_the_index_row_behind_it() {
            const TOPK: u32 = 128;
            let probe = Probe::default();
            let pool = latent_pool();
            attention_decode_selected(
                &probe, bf16(1, 2, HEADS * RANK), bf16(2, 2, HEADS * ROPE), sel(4, 2, TOPK),
                &pool, i32t(7, 2), i32t(8, 2), HEADS, RANK, 0.5, bf16(3, 2, HEADS * RANK),
            )
            .expect("the selected decode enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_naive_paged_selected_bfloat16");
            assert_eq!(f.file, "attn/mla.metal");
            assert_eq!(f.lanes, [HEADS * SIMD, 2, 1]);
            assert_eq!(f.group, [SIMD, 1, 1]);
            assert_eq!(a.len(), 16, "the dense fourteen, then selection and top_k");
            assert_eq!(a[0], ArgValue::Buffer(1));
            assert_eq!(a[4], ArgValue::BufferMut(3));
            assert_eq!(a[5], ArgValue::Buffer(7)); // positions: the causal bound
            assert_eq!(a[6], ArgValue::Buffer(8)); // request_of_token
            assert_eq!(a[13], ArgValue::F32(0.5));
            assert_eq!(a[14], ArgValue::Buffer(4)); // the index row, read-only
            assert_eq!(a[15], ArgValue::I32(TOPK as i32)); // top_k IS its width
        }

        /// Prefill-selected is the same engine over `q.data`.
        #[test]
        fn prefill_selected_shares_the_selected_engine() {
            const TOPK: u32 = 64;
            let probe = Probe::default();
            let pool = latent_pool();
            let q = RaggedTensor { data: bf16(1, 5, HEADS * RANK), indptr: i32t(9, 3) };
            attention_prefill_selected(
                &probe, q, bf16(2, 5, HEADS * ROPE), sel(4, 5, TOPK), &pool,
                i32t(7, 5), i32t(8, 5), HEADS, RANK, 0.5, bf16(3, 5, HEADS * RANK),
            )
            .expect("the selected prefill enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_naive_paged_selected_bfloat16");
            assert_eq!(f.lanes, [HEADS * SIMD, 5, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // q.data, not the indptr
            assert_eq!(a[15], ArgValue::I32(TOPK as i32));
        }

        /// The dense readers keep the dense point and its fourteen seats — the
        /// selected pair did not move anything under them.
        #[test]
        fn the_dense_reader_keeps_its_own_entrypoint() {
            let probe = Probe::default();
            let pool = latent_pool();
            attention_decode(
                &probe, bf16(1, 2, HEADS * RANK), bf16(2, 2, HEADS * ROPE), &pool,
                i32t(7, 2), i32t(8, 2), HEADS, RANK, 0.5, bf16(3, 2, HEADS * RANK),
            )
            .expect("the dense decode enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "mla_naive_paged_bfloat16");
            assert_eq!(a.len(), 14);
        }

        /// A selection carrying a different row count than the reading is
        /// refused by name — `mla.rs`'s guard, mirrored.
        #[test]
        fn a_selection_that_is_not_one_row_per_query_is_refused() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode_selected(
                &probe, bf16(1, 2, HEADS * RANK), bf16(2, 2, HEADS * ROPE), sel(4, 1, 128),
                &pool, i32t(7, 2), i32t(8, 2), HEADS, RANK, 0.5, bf16(3, 2, HEADS * RANK),
            )
            .expect_err("one selection row does not serve two query rows");
            assert!(format!("{why}").contains("one row per query row"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// The geometry refusals bind on the selected point too: the rank the
        /// register strips cannot hold is refused before any launch.
        #[test]
        fn the_selected_point_refuses_the_geometries_the_dense_one_does() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode_selected(
                &probe, bf16(1, 1, HEADS * 500), bf16(2, 1, HEADS * ROPE), sel(4, 1, 8),
                &pool, i32t(7, 1), i32t(8, 1), HEADS, 500, 0.5, bf16(3, 1, HEADS * 500),
            )
            .expect_err("500 is not a multiple of 32");
            assert!(format!("{why}").contains("lane-split"), "{why}");
            assert!(probe.fires().is_empty());
        }

        // ── the sweep semantics, deviceless ────────────────────────────────

        /// **THE -1 PADDED TAIL CONTRIBUTES NO KEY.** `index_topk_paged` pads
        /// a short row with -1; the reader drops those slots rather than
        /// attending key 0 or stopping the row early.
        #[test]
        fn the_padded_tail_is_dropped_not_attended() {
            assert_eq!(selected_sweep(&[0, 2, 5, -1, -1], 8), vec![0, 2, 5]);
            assert_eq!(selected_sweep(&[-1, -1, -1], 8), Vec::<i32>::new());
            assert_eq!(selected_sweep(&[], 8), Vec::<i32>::new());
        }

        /// A -1 in the MIDDLE of the row is skipped, not a stop: the CUDA
        /// reader `continue`s, and the keys behind it are still attended.
        #[test]
        fn a_hole_in_the_row_is_skipped_and_the_walk_continues() {
            assert_eq!(selected_sweep(&[1, -1, 4, -1, 6], 8), vec![1, 4, 6]);
        }

        /// The causal bound still binds: a selected id at or past `j_end` is
        /// dropped, not clamped into the visible range.
        #[test]
        fn the_causal_bound_drops_a_key_the_row_cannot_see() {
            assert_eq!(selected_sweep(&[0, 3, 7, 9], 4), vec![0, 3]);
            assert_eq!(selected_sweep(&[0, 3, 7, 9], 0), Vec::<i32>::new());
            // A row whose whole budget is visible walks all of it, in order.
            assert_eq!(selected_sweep(&[0, 1, 2, 3], 4), vec![0, 1, 2, 3]);
        }

        /// The selection is a SET, not a re-ordering: the streaming softmax
        /// the shader runs over the selected keys equals the batch softmax the
        /// dense reader would produce restricted to exactly those keys.
        #[test]
        fn the_selected_reading_is_the_dense_reading_restricted_to_those_keys() {
            const W: usize = 3;
            // Eight cached keys, one w-wide value strip each.
            let logits: Vec<f32> = vec![0.4, -1.2, 2.0, 0.1, -0.7, 1.5, 0.9, -2.3];
            let values: Vec<f32> = (0..8 * W).map(|i| (i as f32) * 0.25 - 1.0).collect();

            // A row selecting {1, 2, 5, 6} out of a visible [0, 7], padded.
            let row = [1, 2, 5, 6, -1, -1];
            let keys = selected_sweep(&row, 8);
            assert_eq!(keys, vec![1, 2, 5, 6]);

            let picked_logits: Vec<f32> = keys.iter().map(|&j| logits[j as usize]).collect();
            let picked_values: Vec<f32> = keys
                .iter()
                .flat_map(|&j| values[j as usize * W..(j as usize + 1) * W].to_vec())
                .collect();
            let got = flash_reading(&picked_logits, &picked_values, W);

            // The batch reference: softmax over the restricted set, no
            // rescaling, no streaming.
            let m = picked_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = picked_logits.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for i in 0..W {
                let want: f32 = e
                    .iter()
                    .enumerate()
                    .map(|(n, p)| p * picked_values[n * W + i])
                    .sum::<f32>()
                    / z;
                assert!((got[i] - want).abs() < 1e-5, "lane {i}: {} vs {want}", got[i]);
            }
        }

        /// And the whole-budget case closes the loop: a selection naming every
        /// visible key, in key order, reads exactly what the dense sweep reads.
        #[test]
        fn selecting_everything_visible_is_the_dense_reading() {
            const W: usize = 2;
            let logits: Vec<f32> = vec![0.3, 1.1, -0.5, 2.2, 0.0];
            let values: Vec<f32> = (0..5 * W).map(|i| 1.0 - (i as f32) * 0.4).collect();
            let row = [0, 1, 2, 3, 4, -1, -1, -1];
            // The sweep IS the dense sweep: `0..j_end`, in order.
            assert_eq!(selected_sweep(&row, 5), (0..5).collect::<Vec<i32>>());
            let got = flash_reading(&logits, &values, W);
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = logits.iter().map(|s| (s - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for i in 0..W {
                let want: f32 =
                    e.iter().enumerate().map(|(n, p)| p * values[n * W + i]).sum::<f32>() / z;
                assert!((got[i] - want).abs() < 1e-5, "lane {i}: {} vs {want}", got[i]);
            }
            // A row selecting nothing at all reads zero, not NaN: `lsum` never
            // leaves zero and the shader's `inv` guard holds.
            assert_eq!(flash_reading(&[], &[], W), vec![0.0; W]);
        }
    }
}

/// `Index`: the NSA sparse-attention indexer — the metal mirror of
/// `kernels-cuda`'s `attn/index.rs`/`attn/index.cuh`, kernel for kernel.
///
/// This is the top-k SELECTION in front of the sparse attention: the small
/// index key cache is layernormed and roped as it is appended, the index
/// query is roped per head, and `attention.index_topk` scores every visible
/// cached key against it and publishes the `i32` selection row the
/// `attention.mla_*_selected` readers walk.
///
/// The shaders live in `attn/index.metal`. All four IR ops fire:
///
/// | op                              | shader                       |
/// |---------------------------------|------------------------------|
/// | `attention.index_layernorm_rope`| `index_knorm_rope_bfloat16`  |
/// | `attention.index_rope`          | `index_q_rope_bfloat16`      |
/// | `attention.index_kv_append`     | `mla_kv_append_bfloat16`     |
/// | `attention.index_topk`          | `index_topk_paged_bfloat16`  |
///
/// **`index_kv_append` ROUTES TO THE MLA APPENDER, AND THE CUDA TWIN DOES
/// TOO.** `index.rs`'s entry calls `kv::write_mla_to_pages` with both rope
/// arguments `ABSENT` — an index key row is one contiguous plane with no
/// rotated tail to store beside it, which is exactly the latent writer with
/// its rope plane nulled. So this entry calls [`mla::kv_append`] with a
/// zero-wide rope plane rather than shipping a second store that would
/// differ from it in the value of one `int`.
///
/// **`index_topk_mask` IS INTENTIONALLY UNPORTED.** `index.cuh`'s fourth
/// kernel is the dense (unpaged) variant publishing a `u8` mask row. No IR
/// op names it and neither host plane fires it — `index.rs` fires
/// `index_topk_paged` alone — so a metal twin would be a shader with no
/// caller. The shader header states what it would take if a mask-shaped op
/// ever lands.
pub mod index {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, Tensor};

    const FILE: &str = "attn/index.metal";

    /// `index.cuh`'s `kBlock`: the threadgroup the norm and the selection
    /// launch.
    const K_BLOCK: u32 = 256;

    /// `index.cuh`'s `kMaxRopeDim`. On CUDA it is the `float buf[]` each
    /// roping thread stages through; the metal kernels spread the pairs over
    /// the launch and need no buffer, so here it is only the ceiling past
    /// which the authority is undefined — kept so the two planes refuse the
    /// same geometries rather than one of them quietly answering.
    const MAX_ROPE_DIM: u32 = 256;

    /// The threadgroup `index_q_rope` launches: one thread per index head,
    /// rounded up to a whole simdgroup. `index.rs`'s `q_rope_block`.
    #[must_use]
    fn q_rope_block(n_heads: u32) -> u32 {
        (n_heads.div_ceil(32) * 32).max(32)
    }

    /// The rotated prefix this launch may state: an even, nonzero width
    /// inside both the row it rotates and the authority's ceiling.
    fn rotated(op: &'static str, rope_dim: u32, head_dim: u32) -> Result<i32, Error> {
        let rope_dim = nonzero(op, "the rotated prefix this rotation states", rope_dim)?;
        if rope_dim % 2 != 0 {
            return Err(refuse(
                op,
                format!("the rotated prefix {rope_dim} is odd, and this rotation turns pairs"),
            ));
        }
        if rope_dim > head_dim {
            return Err(refuse(
                op,
                format!(
                    "the rotated prefix {rope_dim} is wider than the {head_dim}-wide row it \
                     rotates"
                ),
            ));
        }
        if rope_dim > MAX_ROPE_DIM {
            return Err(refuse(
                op,
                format!(
                    "the rotated prefix {rope_dim} is above the {MAX_ROPE_DIM} this indexer \
                     rotates"
                ),
            ));
        }
        stated(op, rope_dim)
    }

    /// The index pool stores whole key rows contiguously; its token pitch
    /// must spell exactly that. `index.rs`'s `pool_pitch`, minus the HND
    /// clause — the metal pool carries no layout enumerator, and the pitch
    /// question is the same question that clause was asking.
    fn pool_pitch(op: &'static str, pool: &KvPool, row: u32) -> Result<(), Error> {
        if row == 0 {
            return Err(refuse(op, "the index key row is zero-wide"));
        }
        if pool.seq_stride != u64::from(row) {
            return Err(refuse(
                op,
                format!(
                    "the pool's token pitch {} is not the {row}-wide row this index writes",
                    pool.seq_stride
                ),
            ));
        }
        Ok(())
    }

    /// Layernorms the index key row and ropes its head, in place on `k`.
    ///
    /// **A MEAN-SUBTRACTING LAYERNORM WITH A LEARNED BIAS**, not the rms norm
    /// every other entry in this file reaches for — two reductions and an
    /// affine, which is what `index_knorm_rope` is.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_rope(
        ctx: &Ctx<'_>,
        k: Tensor,
        positions: Tensor,
        weight: Tensor,
        bias: Tensor,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_layernorm_rope";
        let entry = dtype_dispatch!(OP, k.dtype, { Bf16 => "index_knorm_rope_bfloat16" });
        debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
        let head_dim = nonzero(OP, "the index key row's width", k.width)?;
        let rope_dim = rotated(OP, rope_dim, head_dim)?;
        let rows = nonzero(OP, "rows", k.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([K_BLOCK, rows, 1], [K_BLOCK, 1, 1])),
            &[
                k.arg_mut(),
                weight.arg(),
                bias.arg(),
                positions.arg(),
                stated(OP, head_dim)?.arg(),
                rope_dim.arg(),
                theta.arg(),
                eps.arg(),
            ],
        )
    }

    /// Ropes the index query's head, per `(row, head)`, in place on `q`.
    pub fn rope(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_rope";
        let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "index_q_rope_bfloat16" });
        debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
        let n_heads = nonzero(OP, "the head count this rotation states", heads)?;
        let head_dim = nonzero(OP, "the head width this rotation states", head_dim)?;
        let rope_dim = rotated(OP, rope_dim, head_dim)?;
        let rows = nonzero(OP, "rows", q.rows)?;
        let block = q_rope_block(n_heads);
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([block, rows, 1], [block, 1, 1])),
            &[
                q.arg_mut(),
                positions.arg(),
                stated(OP, n_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                rope_dim.arg(),
                theta.arg(),
            ],
        )
    }

    /// Appends index key rows into the pool's pages, at the op-named write
    /// tables — the mla latent writer with a null rope plane, which is what
    /// `index.rs` routes to as well.
    ///
    // MENLO-SEAM: as `attention.mla_kv_append`. The op states
    // `write_page`/`write_offset` and on THIS plane the appender reads
    // exactly them, so the seam the CUDA note describes (the stated pair
    // going unread while the writer re-derives the cell) is closed here.
    pub fn kv_append(
        ctx: &Ctx<'_>,
        k: Tensor,
        keys: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_kv_append";
        dtype_dispatch!(OP, k.dtype, { Bf16 => () });
        pool_pitch(OP, keys, k.width)?;
        // A zero-wide rope plane: the store's `rope` bound is 0, so its rope
        // half never reads this handle and never writes the value pages.
        let no_rope = Tensor::new(k.buf, k.rows, 0, k.dtype);
        super::mla::kv_append(ctx, k, no_rope, keys, write_page, write_offset)
    }

    /// Scores `q` against every cached index key visible to its row and
    /// publishes the top-`top_k` cached positions, ascending, `-1`-padded.
    ///
    /// `positions`/`request_of_token` are the fire tables the driver binds
    /// beside the pool, and they are this plane's divergence from the CUDA
    /// twin: `index_topk_paged` re-derives each row's absolute query position
    /// from `qo_indptr` and `kv_last_page_lens`, which the metal pool does not
    /// carry, so the two numbers are READ here the way `mla::flash` and
    /// `pool::attention_lse` already read them.
    ///
    /// `scores` is the per-row working slab the selection writes and then
    /// bisects over — `crate::scratch`'s index role on the shell side, the
    /// process-global scratch on the CUDA one. Its WIDTH is the `score_stride`
    /// the shader clamps `nkeys` against, so a row that can see more keys than
    /// the slab is wide scores its first `score_stride` and no more — which is
    /// `index.cuh`'s own clamp against its own slab, at the same place.
    #[allow(clippy::too_many_arguments)]
    pub fn topk(
        ctx: &Ctx<'_>,
        q: Tensor,
        weights: Tensor,
        keys: &KvPool,
        positions: Tensor,
        request_of_token: Tensor,
        scores: Tensor,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        ratio: u32,
        selection: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.index_topk";
        let entry = dtype_dispatch!(OP, q.dtype, { Bf16 => "index_topk_paged_bfloat16" });
        debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 cached positions");
        debug_assert_eq!(scores.dtype, Dtype::F32, "`{OP}` bisects an f32 score slab");
        debug_assert!(
            positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
            "the fire's position and owning-request tables are i32, one entry per row"
        );
        let heads = nonzero(OP, "the head count this ranking states", heads)?;
        let head_dim = nonzero(OP, "the key width this ranking states", head_dim)?;
        let top_k = nonzero(OP, "the selection budget this ranking states", top_k)?;
        // **THE KEY STRIDE, WHICH IS WHICH CACHED ROWS ARE KEYS.** `1` reads
        // one key per token at its own cell (glm_5); a compressor's ratio
        // reads one key per COMPRESSED BLOCK at the boundary cell
        // `(c+1)*ratio - 1` (dsv4-flash), and the ids published are then
        // compressed rows. Zero is no stride at all and is refused rather
        // than silently taken for one.
        let ratio = nonzero(OP, "the key stride this ranking states", ratio)?;
        if keys.page_size <= 0 {
            return Err(refuse(OP, "the index cache page size is zero"));
        }
        pool_pitch(OP, keys, head_dim)?;
        if q.width != heads.saturating_mul(head_dim) {
            return Err(refuse(
                OP,
                format!(
                    "the {}-wide index query does not divide by the stated head count and \
                     width",
                    q.width
                ),
            ));
        }
        if weights.width != heads {
            return Err(refuse(
                OP,
                "the index head weights are not one per stated head",
            ));
        }
        if selection.width != top_k {
            return Err(refuse(
                OP,
                "the selection this statement allocated is not the budget it stated",
            ));
        }
        let rows = nonzero(OP, "rows", selection.rows)?;
        if scores.rows < rows {
            return Err(refuse(
                OP,
                format!(
                    "the score slab seats {} rows and this ranking launches {rows}",
                    scores.rows
                ),
            ));
        }
        let stride = nonzero(OP, "the score slab's key stride", scores.width)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([K_BLOCK, rows, 1], [K_BLOCK, 1, 1])),
            &[
                q.arg(),
                weights.arg(),
                keys.keys.arg(),
                positions.arg(),
                request_of_token.arg(),
                keys.page_indices.arg(),
                keys.page_indptr.arg(),
                scores.arg_mut(),
                selection.arg_mut(),
                stated(OP, heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                keys.page_size.arg(),
                stated(OP, stride)?.arg(),
                stated(OP, top_k)?.arg(),
                stated(OP, ratio)?.arg(),
            ],
        )
    }

    /// The bisection this family selects by, in host arithmetic — the
    /// deviceless pin on the ONE part of `index_topk_paged` that is an
    /// algorithm rather than a launch.
    ///
    /// **IT IS HERE SO THE SEMANTICS CAN BE TESTED WITHOUT A GPU.** Nothing on
    /// the fire path calls it: the selection runs on the device, in the
    /// shader, over scores the device wrote. What the tests below ask of it is
    /// what no `Probe` can ask of a `Fire` — that 40 halvings of `[min, max]`
    /// counting `>= mid`, with `lo = mid` when the count still exceeds the
    /// budget and `hi = mid` when it does not, followed by an ascending walk
    /// taking the first `topk` keys at or above `hi`, selects the set the CUDA
    /// kernel selects. Every line below has a line above it in
    /// `attn/index.metal` and in `index.cuh`, and the iteration count is
    /// contract rather than tolerance.
    #[must_use]
    pub fn bisect_select(scores: &[f32], topk: usize) -> Vec<i32> {
        let nkeys = scores.len();
        if nkeys <= topk {
            return (0..topk)
                .map(|n| if n < nkeys { n as i32 } else { -1 })
                .collect();
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for s in scores {
            lo = lo.min(*s);
            hi = hi.max(*s);
        }
        let mut thr = hi;
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let cnt = scores.iter().filter(|s| **s >= mid).count();
            if cnt > topk {
                lo = mid;
            } else {
                hi = mid;
            }
            thr = hi;
        }
        let mut out = Vec::with_capacity(topk);
        for (j, s) in scores.iter().enumerate() {
            if out.len() == topk {
                break;
            }
            if *s >= thr {
                out.push(j as i32);
            }
        }
        out.resize(topk, -1);
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::encode::ArgValue;
        use crate::probe::Probe;

        /// dsv4-flash's indexer geometry: 64 index heads of 128 lanes, a 512
        /// selection budget.
        const HEADS: u32 = 64;
        const DIM: u32 = 128;
        const TOPK: u32 = 512;

        fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
            Tensor::new(buf, rows, width, Dtype::Bf16)
        }
        fn i32t(buf: u32, rows: u32, width: u32) -> Tensor {
            Tensor::new(buf, rows, width, Dtype::I32)
        }
        fn u32t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::U32)
        }
        fn f32t(buf: u32, rows: u32, width: u32) -> Tensor {
            Tensor::new(buf, rows, width, Dtype::F32)
        }

        /// The index cache: whole `DIM`-wide key rows, one per cached token.
        fn index_pool() -> KvPool {
            KvPool {
                keys: bf16(50, 4096, DIM),
                values: bf16(51, 4096, DIM),
                page_indices: u32t(52, 64),
                page_indptr: u32t(53, 8),
                page_size: 16,
                seq_stride: u64::from(DIM),
                head_stride: u64::from(DIM),
            }
        }

        /// The norm is one threadgroup of 256 per cached key row; the row is
        /// the mutated plane and `w`/`b` are the learned affine.
        #[test]
        fn layernorm_rope_is_a_threadgroup_per_key_row() {
            let probe = Probe::default();
            layernorm_rope(&probe, bf16(1, 6, DIM), i32t(2, 6, 1), bf16(3, 1, DIM), bf16(4, 1, DIM), 1e-6, 64, 10_000.0)
                .expect("the index norm enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.file, "attn/index.metal");
            assert_eq!(f.entrypoint, "index_knorm_rope_bfloat16");
            assert_eq!(f.lanes, [K_BLOCK, 6, 1]);
            assert_eq!(f.group, [K_BLOCK, 1, 1]);
            assert_eq!(a[0], ArgValue::BufferMut(1)); // idx_k, in place
            assert_eq!(a[1], ArgValue::Buffer(3)); // w
            assert_eq!(a[2], ArgValue::Buffer(4)); // b
            assert_eq!(a[3], ArgValue::Buffer(2)); // positions
            assert_eq!(a[4], ArgValue::I32(DIM as i32));
            assert_eq!(a[5], ArgValue::I32(64)); // rope_dim
            assert_eq!(a[6], ArgValue::F32(10_000.0));
            assert_eq!(a[7], ArgValue::F32(1e-6));
        }

        /// The query rotation is one thread per `(row, head)`, the block
        /// rounded up to a whole simdgroup — `index.rs`'s `q_rope_block`.
        #[test]
        fn rope_is_a_thread_per_row_head_on_a_simd_rounded_block() {
            let probe = Probe::default();
            rope(&probe, bf16(1, 3, HEADS * DIM), i32t(2, 3, 1), HEADS, DIM, 64, 10_000.0)
                .expect("the index query rotation enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "index_q_rope_bfloat16");
            assert_eq!(f.lanes, [HEADS, 3, 1]);
            assert_eq!(f.group, [HEADS, 1, 1]);
            assert_eq!(a[0], ArgValue::BufferMut(1)); // idx_q, in place
            assert_eq!(a[1], ArgValue::Buffer(2)); // positions
            assert_eq!(a[2], ArgValue::I32(HEADS as i32));
            assert_eq!(a[3], ArgValue::I32(DIM as i32));
            assert_eq!(a[4], ArgValue::I32(64));
            assert_eq!(a[5], ArgValue::F32(10_000.0));
        }

        /// A head count that is not a whole simdgroup still launches one:
        /// 40 heads round to a 64-thread block, and the shader drops the tail.
        #[test]
        fn rope_rounds_a_ragged_head_count_up_to_a_simdgroup() {
            assert_eq!(q_rope_block(40), 64);
            assert_eq!(q_rope_block(64), 64);
            assert_eq!(q_rope_block(1), 32);
            let probe = Probe::default();
            rope(&probe, bf16(1, 2, 40 * DIM), i32t(2, 2, 1), 40, DIM, 64, 10_000.0)
                .expect("a ragged head count enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.lanes, [64, 2, 1]);
            assert_eq!(f.group, [64, 1, 1]);
            assert_eq!(a[2], ArgValue::I32(40));
        }

        /// The append is the mla latent writer with a null rope plane: the
        /// index row lands in the key pages, the value pages go untouched
        /// because the store's rope bound is zero.
        #[test]
        fn kv_append_is_the_latent_writer_with_a_null_rope_plane() {
            let probe = Probe::default();
            let pool = index_pool();
            kv_append(&probe, bf16(1, 4, DIM), &pool, u32t(7, 4), u32t(8, 4))
                .expect("the index append enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.file, "attn/mla.metal");
            assert_eq!(f.entrypoint, "mla_kv_append_bfloat16");
            assert_eq!(f.lanes, [DIM, 4, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // the index key rows
            assert_eq!(a[1], ArgValue::Buffer(1)); // the null rope plane
            assert_eq!(a[2], ArgValue::BufferMut(50)); // key pages
            assert_eq!(a[4], ArgValue::Buffer(7)); // write_page
            assert_eq!(a[5], ArgValue::Buffer(8)); // write_offset
            assert_eq!(a[7], ArgValue::I32(DIM as i32)); // kv_lora = the row
            assert_eq!(a[8], ArgValue::I32(0)); // rope = 0: nothing else stored
        }

        /// A pool whose token pitch is not the index row refuses by name — the
        /// index cache stores whole rows contiguously or it stores them
        /// scattered.
        #[test]
        fn kv_append_refuses_a_pool_that_is_not_one_row_per_token() {
            let probe = Probe::default();
            let pool = KvPool { seq_stride: u64::from(DIM) * 2, ..index_pool() };
            let why = kv_append(&probe, bf16(1, 4, DIM), &pool, u32t(7, 4), u32t(8, 4))
                .expect_err("a doubled pitch is not this row");
            assert!(format!("{why}").contains("token pitch"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// The selection at the served geometry — 512 of 64x128 — is one
        /// threadgroup of 256 per query row, with the fire tables in the seats
        /// the CUDA twin fills from `qo_indptr`/`kv_last_page_lens`.
        #[test]
        fn topk_is_a_threadgroup_per_query_row_at_the_dsv4_geometry() {
            let probe = Probe::default();
            let pool = index_pool();
            topk(
                &probe, bf16(1, 3, HEADS * DIM), bf16(2, 3, HEADS), &pool,
                i32t(9, 3, 1), i32t(10, 3, 1), f32t(11, 3, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 3, TOPK),
            )
            .expect("the selection enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.file, "attn/index.metal");
            assert_eq!(f.entrypoint, "index_topk_paged_bfloat16");
            assert_eq!(f.lanes, [K_BLOCK, 3, 1]);
            assert_eq!(f.group, [K_BLOCK, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // idx_q
            assert_eq!(a[1], ArgValue::Buffer(2)); // idx_w
            assert_eq!(a[2], ArgValue::Buffer(50)); // the index key pages
            assert_eq!(a[3], ArgValue::Buffer(9)); // positions: the causal bound
            assert_eq!(a[4], ArgValue::Buffer(10)); // request_of_token
            assert_eq!(a[5], ArgValue::Buffer(52)); // page_indices
            assert_eq!(a[6], ArgValue::Buffer(53)); // page_indptr
            assert_eq!(a[7], ArgValue::BufferMut(11)); // the score slab
            assert_eq!(a[8], ArgValue::BufferMut(12)); // selection
            assert_eq!(a[9], ArgValue::I32(HEADS as i32));
            assert_eq!(a[10], ArgValue::I32(DIM as i32));
            assert_eq!(a[11], ArgValue::I32(16)); // page_size
            assert_eq!(a[12], ArgValue::I32(8192)); // score_stride = the slab width
            assert_eq!(a[13], ArgValue::I32(TOPK as i32));
            // glm_5's stride: one key per token, so the cell IS the id.
            assert_eq!(a[14], ArgValue::I32(1));
        }

        /// **THE dsv4 STRIDE.** The same launch with `ratio = 4`: the shader
        /// then sees `(pos+1)/4` keys and reads key `c` at the boundary cell
        /// `(c+1)*4 - 1`, which is where `pool_kv_append` put the indexer
        /// compressor's pooled entry. Only the last argument moves.
        #[test]
        fn topk_states_the_compressors_stride_when_the_keys_are_pooled() {
            let probe = Probe::default();
            let pool = index_pool();
            topk(
                &probe, bf16(1, 3, HEADS * DIM), bf16(2, 3, HEADS), &pool,
                i32t(9, 3, 1), i32t(10, 3, 1), f32t(11, 3, 8192),
                HEADS, DIM, TOPK, 4, i32t(12, 3, TOPK),
            )
            .expect("the selection enqueues at the compressor's stride");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "index_topk_paged_bfloat16");
            assert_eq!(a[14], ArgValue::I32(4));
        }

        /// A zero stride is no stride at all — refused by name rather than
        /// taken for one, which is what a `/ 0` in the shader would be.
        #[test]
        fn topk_refuses_a_zero_key_stride() {
            let probe = Probe::default();
            let pool = index_pool();
            let why = topk(
                &probe, bf16(1, 1, HEADS * DIM), bf16(2, 1, HEADS), &pool,
                i32t(9, 1, 1), i32t(10, 1, 1), f32t(11, 1, 8192),
                HEADS, DIM, TOPK, 0, i32t(12, 1, TOPK),
            )
            .expect_err("zero is not a key stride");
            assert!(format!("{why}").contains("key stride"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// A query row that does not divide by the stated head count and
        /// width, head weights that are not one per head, and a selection
        /// that is not the stated budget: `index.rs`'s three shape refusals,
        /// each by name and none launched.
        #[test]
        fn topk_refuses_the_three_shapes_its_cuda_twin_refuses() {
            let probe = Probe::default();
            let pool = index_pool();
            let bad_q = topk(
                &probe, bf16(1, 1, HEADS * DIM + 8), bf16(2, 1, HEADS), &pool,
                i32t(9, 1, 1), i32t(10, 1, 1), f32t(11, 1, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 1, TOPK),
            )
            .expect_err("the query does not divide");
            assert!(format!("{bad_q}").contains("does not divide"), "{bad_q}");

            let bad_w = topk(
                &probe, bf16(1, 1, HEADS * DIM), bf16(2, 1, HEADS - 1), &pool,
                i32t(9, 1, 1), i32t(10, 1, 1), f32t(11, 1, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 1, TOPK),
            )
            .expect_err("the weights are not one per head");
            assert!(format!("{bad_w}").contains("one per stated head"), "{bad_w}");

            let bad_sel = topk(
                &probe, bf16(1, 1, HEADS * DIM), bf16(2, 1, HEADS), &pool,
                i32t(9, 1, 1), i32t(10, 1, 1), f32t(11, 1, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 1, TOPK - 1),
            )
            .expect_err("the selection is not the budget");
            assert!(format!("{bad_sel}").contains("budget it stated"), "{bad_sel}");

            assert!(probe.fires().is_empty());
        }

        /// The score slab is the one thing this plane has that the CUDA twin
        /// allocates per fire, so a slab too short for the launch is refused
        /// with both numbers rather than overrun.
        #[test]
        fn topk_refuses_a_score_slab_shorter_than_the_launch() {
            let probe = Probe::default();
            let pool = index_pool();
            let why = topk(
                &probe, bf16(1, 8, HEADS * DIM), bf16(2, 8, HEADS), &pool,
                i32t(9, 8, 1), i32t(10, 8, 1), f32t(11, 4, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 8, TOPK),
            )
            .expect_err("four slab rows do not seat eight query rows");
            assert!(format!("{why}").contains("seats 4 rows"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// The rotated prefix is refused above `kMaxRopeDim`, odd, and wider
        /// than the row it turns — on both rotations.
        #[test]
        fn a_rotation_past_the_authority_is_refused() {
            let probe = Probe::default();
            let over = layernorm_rope(&probe, bf16(1, 1, 512), i32t(2, 1, 1), bf16(3, 1, 512), bf16(4, 1, 512), 1e-6, 384, 10_000.0)
                .expect_err("384 is past kMaxRopeDim");
            assert!(format!("{over}").contains("above the 256"), "{over}");
            let odd = rope(&probe, bf16(1, 1, HEADS * DIM), i32t(2, 1, 1), HEADS, DIM, 63, 10_000.0)
                .expect_err("an odd prefix has no last pair");
            assert!(format!("{odd}").contains("odd"), "{odd}");
            let wide = rope(&probe, bf16(1, 1, HEADS * DIM), i32t(2, 1, 1), HEADS, DIM, 256, 10_000.0)
                .expect_err("256 is wider than the 128-wide head");
            assert!(format!("{wide}").contains("wider than"), "{wide}");
            assert!(probe.fires().is_empty());
        }

        /// A pool of a dtype no point is stamped for is refused, not
        /// mis-fired: bf16 is the one the indexer ships.
        #[test]
        fn a_dtype_no_point_is_stamped_for_is_refused() {
            let probe = Probe::default();
            let pool = index_pool();
            let why = topk(
                &probe, Tensor::new(1, 1, HEADS * DIM, Dtype::F32), bf16(2, 1, HEADS), &pool,
                i32t(9, 1, 1), i32t(10, 1, 1), f32t(11, 1, 8192),
                HEADS, DIM, TOPK, 1, i32t(12, 1, TOPK),
            )
            .expect_err("f32 index queries are not stamped");
            assert!(
                matches!(why, Error::DtypeUnsupported { op: "attention.index_topk", .. }),
                "{why}"
            );
            assert!(probe.fires().is_empty());
        }

        // ── the selection semantics, pinned without a device ────────────────

        /// Fewer visible keys than the budget: the selection is the identity
        /// over what exists, and the tail is `-1` — the early path, taken
        /// before the bisection runs at all.
        #[test]
        fn a_row_that_sees_fewer_keys_than_the_budget_selects_all_of_them() {
            assert_eq!(bisect_select(&[3.0, 1.0, 2.0], 5), vec![0, 1, 2, -1, -1]);
            assert_eq!(bisect_select(&[3.0, 1.0, 2.0], 3), vec![0, 1, 2]);
            assert_eq!(bisect_select(&[], 2), vec![-1, -1]);
        }

        /// The bisection picks the budget's worth of largest scores, and
        /// publishes them in ASCENDING KEY ORDER — the selection is a set of
        /// positions, not a ranking.
        #[test]
        fn the_bisection_selects_the_largest_scores_in_key_order() {
            let scores = [0.1f32, 9.0, 0.2, 8.0, 0.3, 7.0, 0.4, 6.0];
            assert_eq!(bisect_select(&scores, 3), vec![1, 3, 5]);
            assert_eq!(bisect_select(&scores, 1), vec![1]);
            let mut all = bisect_select(&scores, 4);
            all.sort_unstable();
            assert_eq!(all, vec![1, 3, 5, 7]);
        }

        /// **TIES AT THE THRESHOLD GO TO THE EARLIER KEY.** Eight equal
        /// scores and a budget of three: the walk is ascending and stops at
        /// the budget, so it is keys 0, 1, 2 — never a later three, and never
        /// more than the budget however many clear the threshold.
        #[test]
        fn a_tie_at_the_threshold_is_broken_by_position() {
            assert_eq!(bisect_select(&[1.0; 8], 3), vec![0, 1, 2]);
            let scores = [5.0f32, 1.0, 5.0, 1.0, 5.0, 1.0];
            assert_eq!(bisect_select(&scores, 2), vec![0, 2]);
            // Every score identical AND the budget met exactly: the threshold
            // admits all six, and the collect still stops at four.
            assert_eq!(bisect_select(&[2.5; 6], 4), vec![0, 1, 2, 3]);
        }

        /// Negative and zero scores are ordinary: the relu is inside the
        /// per-head dot, so a row's total can still be zero for every head
        /// that contributes nothing, and the bisection brackets `[min, max]`
        /// wherever they are.
        #[test]
        fn the_bracket_is_the_row_s_own_range_wherever_it_sits() {
            let scores = [-4.0f32, -1.0, -3.0, -2.0];
            assert_eq!(bisect_select(&scores, 2), vec![1, 3]);
            assert_eq!(bisect_select(&[0.0, 0.0, 1e-30, 0.0], 1), vec![2]);
        }

        /// **FORTY HALVINGS, AND THE NUMBER IS THE CONTRACT.** A budget of
        /// `k` over distinct scores selects exactly `k` real keys and pads
        /// nothing — which is only true if the bisection has converged past
        /// every gap in the row. Run it over a long ramp, where the gap
        /// between neighbouring scores is small enough that a shorter
        /// bisection would admit the wrong count.
        #[test]
        fn forty_halvings_converge_past_every_gap_in_a_long_row() {
            let scores: Vec<f32> = (0..4096).map(|j| j as f32 * 1e-4).collect();
            for topk in [1usize, 7, 512, 4095] {
                let picked = bisect_select(&scores, topk);
                assert_eq!(picked.len(), topk);
                assert!(picked.iter().all(|j| *j >= 0), "budget {topk} padded");
                // The top `topk` of a strictly ascending ramp is its tail.
                assert_eq!(picked[0], (4096 - topk) as i32);
                assert_eq!(*picked.last().expect("nonempty"), 4095);
            }
        }
    }
}

/// `Pool`: pooled (compressed) attention — the dsv4 compressor's KV-time-axis
/// pooling, ported organ-for-organ from `kernels-cuda/kernels/attn/pool.cuh`.
///
/// The paged shaders live in `attn/pool.metal`. All five ops are wired here:
/// the two boundary detectors, the gated softmax pool ([`gather`]), the
/// compressed-cache store, and the log-sum-exp reader over the compressed
/// entries. Four of them read only what their IR op names; the fifth is the
/// family's one MENLO-SEAM, and it is spelled out on [`gather`] itself.
pub mod pool {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/pool.metal";

    /// The flat block the boundary detectors launch (`pool.cuh`'s META_BLOCK).
    const META_BLOCK: u32 = 128;

    /// The threadgroup width the compressed-cache flash reader launches
    /// (`pool.cuh`'s ATTN_BLOCK).
    const ATTN_BLOCK: u32 = 128;

    /// The widest head the flash reader holds in its q tile — the shader's
    /// `POOL_HEAD_MAX` threadgroup array bound.
    const POOL_HEAD_MAX: u32 = 512;

    /// `2` for the overlapping `2*ratio` window of the ratio-4 compressor,
    /// `1` otherwise — the twin of `pool.rs`'s `compressor_coff`.
    ///
    /// **DERIVED HERE AND NOT STATED BY THE CALLER**, which is the CUDA
    /// entry's choice for the CUDA entry's reason: `coff` is a function of the
    /// ratio the op already states, so a caller that could pass a third answer
    /// could bind a window the state slabs are not laid out for.
    const fn compressor_coff(ratio: u32) -> i32 {
        if ratio == 4 { 2 } else { 1 }
    }

    fn boundary_tables(op: &'static str, boundary_pos: &Tensor, boundary_req: &Tensor) {
        debug_assert_eq!(boundary_pos.dtype, Dtype::I32, "`{op}` reads i32 boundary positions");
        debug_assert_eq!(boundary_req.dtype, Dtype::I32, "`{op}` reads i32 boundary requests");
        debug_assert_eq!(
            boundary_pos.rows, boundary_req.rows,
            "`{op}`'s boundary tables are one entry per token row"
        );
    }

    /// The third boundary column: the compressed row's rope position, one
    /// entry per token row like the two beside it.
    fn boundary_rope_table(op: &'static str, boundary_pos: &Tensor, boundary_rope: &Tensor) {
        debug_assert_eq!(
            boundary_rope.dtype,
            Dtype::I32,
            "`{op}` writes i32 compressed-row rope positions"
        );
        debug_assert_eq!(
            boundary_pos.rows, boundary_rope.rows,
            "`{op}`'s rope column is one entry per token row"
        );
    }

    /// Marks which decode rows close a pooling boundary. `row_valid` is the
    /// CUDA-graph padding mask, an op-named `u8` input.
    ///
    /// `boundary_rope` is the compressed row's own rope position — the
    /// block's FIRST token `(p / ratio) · ratio`, not `boundary_pos`'s LAST
    /// one — which the CUDA twin has always computed and which had nowhere to
    /// land on this plane until the compressor fired. See the shader's note.
    pub fn boundary_decode(
        ctx: &Ctx<'_>,
        positions: Tensor,
        row_valid: Tensor,
        ratio: u32,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        boundary_rope: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_boundary_decode";
        boundary_tables(OP, &boundary_pos, &boundary_req);
        boundary_rope_table(OP, &boundary_pos, &boundary_rope);
        let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        ctx.fire(
            Fire::at(FILE, "pool_boundary_decode")
                .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
            &[
                positions.arg(),
                boundary_pos.arg_mut(),
                boundary_req.arg_mut(),
                boundary_rope.arg_mut(),
                n.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }

    /// The prefill twin: boundaries within each request's ragged span, the
    /// owning request a binary search over the fire's `qo_indptr`.
    #[allow(clippy::too_many_arguments)]
    pub fn boundary_prefill(
        ctx: &Ctx<'_>,
        positions: RaggedTensor,
        row_valid: Tensor,
        ratio: u32,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        boundary_rope: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_boundary_prefill";
        boundary_tables(OP, &boundary_pos, &boundary_req);
        boundary_rope_table(OP, &boundary_pos, &boundary_rope);
        let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        let num_requests = stated(
            OP,
            nonzero(OP, "requests", positions.indptr.rows.saturating_sub(1))?,
        )?;
        ctx.fire(
            Fire::at(FILE, "pool_boundary_prefill")
                .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
            &[
                positions.data.arg(),
                positions.indptr.arg(),
                boundary_pos.arg_mut(),
                boundary_req.arg_mut(),
                boundary_rope.arg_mut(),
                n.arg(),
                num_requests.arg(),
                ratio.arg(),
                row_valid.arg(),
            ],
        )
    }

    /// **THE ROLLING STATE'S WRITER.** `kv` is the compressor's `wkv · x` and
    /// `score` its `wgate · x`, both `[rows, coff · head_dim]`; each row is
    /// scattered into the cell `write_page`/`write_offset` name for it — the
    /// SOURCE cache's own slot, which is the cell the latent appender writes
    /// in the same fire.
    ///
    /// **THIS IS THE OP THAT MAKES [`gather`] POOL SOMETHING.** The two state
    /// slabs are a seam the shell owns and no IR value names, and until this
    /// entry existed nothing wrote a byte of either: the gather fired, read
    /// zeros, and the compressor's four checkpoint planes were interned. The
    /// slabs stay seam arguments here for the same reason they are there —
    /// they are addressed by the cache's cell and not by the fire's row.
    #[allow(clippy::too_many_arguments)]
    pub fn state_write(
        ctx: &Ctx<'_>,
        kv: Tensor,
        score: Tensor,
        pages: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_state_write";
        let entry = dtype_dispatch!(OP, kv.dtype, { Bf16 => "pool_state_write_bfloat16" });
        debug_assert_eq!(
            score.dtype, kv.dtype,
            "`{OP}` writes both projections in one element"
        );
        debug_assert!(
            state_kv.dtype == kv.dtype && state_score.dtype == kv.dtype,
            "`{OP}` lands the state plane in the projections' own element"
        );
        debug_assert!(
            write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
            "the write tables are u32: one destination page and one in-page slot per row"
        );
        if pages.page_size <= 0 {
            return Err(refuse(OP, "the source pool's page size is zero"));
        }
        let head_dim = nonzero(OP, "the head width this compressor states", head_dim)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let coff = compressor_coff(ratio);
        // The columns one state row holds for THIS layer — what the gather
        // over the same ratio reads back.
        let width = head_dim.saturating_mul(coff.unsigned_abs());
        if kv.width != width || score.width != width {
            return Err(refuse(
                OP,
                format!(
                    "a ratio-{ratio} compressor projects coff {coff} x head width \
                     {head_dim} = {width} columns; the pair handed over is {} and {}",
                    kv.width, score.width
                ),
            ));
        }
        debug_assert_eq!(
            score.rows, kv.rows,
            "the two projections are one row per token row"
        );
        // The plane's row, which is not always this layer's `width`: see
        // [`gather`]'s note on one artifact holding two ratios.
        let pitch = state_kv.width;
        if state_score.width != pitch {
            return Err(refuse(
                OP,
                format!(
                    "`state_kv` is {pitch} wide and `state_score` is {} — the two state \
                     slabs are one plane laid at one pitch",
                    state_score.width
                ),
            ));
        }
        if pitch < width {
            return Err(refuse(
                OP,
                format!(
                    "the state slabs are {pitch} wide and this compressor writes coff \
                     {coff} x head width {head_dim} = {width} columns of every row"
                ),
            ));
        }
        let rows = nonzero(OP, "rows", kv.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([width, rows, 1], [width.min(256), 1, 1])),
            &[
                kv.arg(),
                score.arg(),
                state_kv.arg_mut(),
                state_score.arg_mut(),
                write_page.arg(),
                write_offset.arg(),
                stated(OP, width)?.arg(),
                pages.page_size.arg(),
                stated(OP, pitch)?.arg(),
            ],
        )
    }

    /// Pools the closing window out of the rolling compressor state into
    /// per-boundary entries — the learned gated softmax pool.
    ///
    /// One thread per `(entry, head-dim lane)`; each walks its `coff * ratio`
    /// window serially, so the whole gate is one launch and no threadgroup
    /// memory.
    ///
    // MENLO-SEAM: the rolling compressor state (`state_kv`, `state_score`)
    // has no IR seat; the engine binds the slabs it staged for the pooled
    // space's SOURCE cache. `ape` closed its half of the seam — it is a
    // checkpoint plane and takes an operand.
    ///
    /// **THE TWO STATE SLABS ARE SEAM ARGUMENTS AND NOT OPERANDS.**
    /// `state_kv` (the rolling kv window, the `wkv` projection's output) and
    /// `state_score` (the rolling gate logits, `wgate`'s) are addressed by the
    /// SAME paged slot the `pages` cache is, at a row pitch of
    /// `coff * head_dim` — so a slab is `[the source pool's cells,
    /// coff * head_dim]` and not a fire-shaped rectangle. Neither is named by
    /// `attention.pool_gather`; the CUDA twin binds them off fire state
    /// (`Run::slabs()`) and engine-metal off its scratch reservation
    /// (`crate::scratch`'s pool role, the `index` role's shape).
    /// [`state_write`] is what puts numbers in them.
    ///
    /// **`ape` IS AN OPERAND**, and the third slab it used to be counted
    /// beside. It is a checkpoint WEIGHT — the compressor's intra-block
    /// absolute-position plane, `[ratio, coff * head_dim]` f32 — not shell
    /// scratch, so it took an IR seat (`Attention::PoolGather.ape`) rather
    /// than a staged rectangle. It stays an `Option` because the CUDA shader
    /// keys the position fold on `ape != nullptr` and Metal has no null
    /// buffer: `None` binds `state_score`'s handle into the unread seat and
    /// states `has_ape = 0`, which is the same path by a different spelling,
    /// and is what a parameter-free mean pool passes.
    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        ctx: &Ctx<'_>,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        pages: &KvPool,
        head_dim: u32,
        ratio: u32,
        state_kv: Tensor,
        state_score: Tensor,
        ape: Option<Tensor>,
        entries: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_gather";
        let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "pool_gather_paged_bfloat16" });
        boundary_tables(OP, &boundary_pos, &boundary_req);
        debug_assert_eq!(
            state_kv.dtype, entries.dtype,
            "`{OP}` pools the state plane into an entry of its own element"
        );
        debug_assert_eq!(
            state_score.dtype, entries.dtype,
            "`{OP}` reads the gate logits in the state plane's element"
        );
        if pages.page_size <= 0 {
            return Err(refuse(OP, "the pooled space's page size is zero"));
        }
        let head_dim = nonzero(OP, "the head width this gather states", head_dim)?;
        if head_dim != entries.width {
            return Err(refuse(
                OP,
                format!(
                    "the stated head width {head_dim} is not the {}-wide entry it sized",
                    entries.width
                ),
            ));
        }
        if head_dim > POOL_HEAD_MAX {
            return Err(refuse(
                OP,
                format!(
                    "the head width {head_dim} is above the {POOL_HEAD_MAX} this pool \
                     launches as one threadgroup"
                ),
            ));
        }
        let rows = nonzero(OP, "rows", boundary_pos.rows)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let coff = compressor_coff(ratio);
        // The columns this gather reads out of one state row.
        let width = head_dim.saturating_mul(coff.unsigned_abs());
        // **AND THE ROW PITCH IS THE SLAB'S, WHICH IS NOT ALWAYS THIS
        // GATHER'S WIDTH.** One artifact can hold pooled layers at two ratios
        // — dsv4-flash carries ratio 4 (coff 2) and ratio 128 (coff 1) in the
        // same tower — and the reservation lays ONE plane for all of them, at
        // the widest pitch any of its gathers states
        // (`engine_metal::scratch::pool_state`). So a narrower gather strides
        // by the plane's row and reads its own `coff x head_dim` columns
        // inside it. What is still refused is a slab NARROWER than the columns
        // read, which is a launch reading somebody else's cells, and a pair
        // that disagrees with each other, which is two pitches for one plane.
        let pitch = state_kv.width;
        if state_score.width != pitch {
            return Err(refuse(
                OP,
                format!(
                    "`state_kv` is {pitch} wide and `state_score` is {} — the two state \
                     slabs are one plane laid at one pitch",
                    state_score.width
                ),
            ));
        }
        if pitch < width {
            return Err(refuse(
                OP,
                format!(
                    "the state slabs are {pitch} wide and this gather reads coff {coff} x \
                     head width {head_dim} = {width} columns of every row"
                ),
            ));
        }
        if let Some(ape) = ape {
            if ape.dtype != Dtype::F32 {
                return Err(refuse(OP, "the absolute-position plane is read as f32"));
            }
            if ape.width != width || ape.rows != ratio {
                return Err(refuse(
                    OP,
                    format!(
                        "the absolute-position plane is {} x {} and this gather reads it at \
                         [ratio {ratio}, {width}]",
                        ape.rows, ape.width
                    ),
                ));
            }
        }
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([head_dim, rows, 1], [head_dim, 1, 1])),
            &[
                state_kv.arg(),
                state_score.arg(),
                // The unread seat when no plane was staged — see the entry
                // note. It is a bound buffer and never a read.
                ape.unwrap_or(state_score).arg(),
                boundary_pos.arg(),
                boundary_req.arg(),
                pages.page_indices.arg(),
                pages.page_indptr.arg(),
                entries.arg_mut(),
                stated(OP, head_dim)?.arg(),
                stated(OP, ratio)?.arg(),
                coff.arg(),
                pages.page_size.arg(),
                i32::from(ape.is_some()).arg(),
                stated(OP, pitch)?.arg(),
            ],
        )
    }

    /// Stores each pooled entry into its cell of the compressed cache. One
    /// threadgroup lane per `(entry, head-dim)`; a masked boundary writes
    /// nothing. `write_page`/`write_offset` are the op's stated write
    /// descriptors — the store still re-derives its cell from the boundary
    /// tables and the pool's page tables, the same MENLO-SEAM the CUDA twin
    /// carries, so the stated pair goes unread.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        ctx: &Ctx<'_>,
        entries: Tensor,
        boundary_pos: Tensor,
        boundary_req: Tensor,
        pool: &KvPool,
        write_page: Tensor,
        write_offset: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_kv_append";
        let _ = (write_page, write_offset);
        let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "pool_store_entries_bfloat16" });
        boundary_tables(OP, &boundary_pos, &boundary_req);
        if pool.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let head_dim = nonzero(OP, "the pooled entry's width", entries.width)?;
        if head_dim > POOL_HEAD_MAX {
            return Err(refuse(
                OP,
                format!("the pooled entry width {head_dim} is above the {POOL_HEAD_MAX} this store launches as one threadgroup"),
            ));
        }
        let rows = nonzero(OP, "rows", entries.rows)?;
        ctx.fire(
            Fire::at(FILE, entry).apply(Grid::of([head_dim, rows, 1], [head_dim, 1, 1])),
            &[
                entries.arg(),
                pool.keys.arg_mut(),
                boundary_pos.arg(),
                boundary_req.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                stated(OP, head_dim)?.arg(),
                pool.page_size.arg(),
            ],
        )
    }

    /// Attention over the compressed entries, publishing the base-2
    /// log-sum-exp plane a later `attention.merge_lse` folds against the dense
    /// pass. One threadgroup per `(query row, query head)`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_lse(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        request_of_token: Tensor,
        entries: &KvPool,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Tensor,
        lse: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_lse";
        dtype_dispatch!(OP, q.dtype, { Bf16 => () });
        debug_assert_eq!(lse.dtype, Dtype::F32, "`{OP}` lands an f32 log-sum-exp plane");
        debug_assert_eq!(
            request_of_token.dtype, Dtype::I32,
            "`{OP}` reads an i32 owning request per token"
        );
        if entries.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let num_q_heads = nonzero(OP, "heads", heads)?;
        let head_dim = nonzero(OP, "the head width", head_dim)?;
        if head_dim > POOL_HEAD_MAX {
            return Err(refuse(
                OP,
                format!("the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in threadgroup memory"),
            ));
        }
        let rows = nonzero(OP, "rows", o.rows)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        ctx.fire(
            Fire::at(FILE, "pool_lse_paged")
                .apply(Grid::of([ATTN_BLOCK, rows, num_q_heads], [ATTN_BLOCK, 1, 1])),
            &[
                q.arg(),
                entries.keys.arg(),
                o.arg_mut(),
                lse.arg_mut(),
                positions.arg(),
                entries.page_indices.arg(),
                entries.page_indptr.arg(),
                request_of_token.arg(),
                stated(OP, num_q_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                ratio.arg(),
                entries.page_size.arg(),
                sm_scale.arg(),
            ],
        )
    }

    /// [`attention_lse`] over the compressed rows `attention.index_topk`
    /// chose — the NSA fine branch, and the ONE reader this family was
    /// missing.
    ///
    /// **THE SELECTED BRANCH IS THE COMPRESSED ONE, NARROWED.** The
    /// reference oracle attends `concat(the 128-wide sliding window over the
    /// per-token latent, EVERY visible compressed row)` under one softmax
    /// with the per-head sink in the denominator; pie computes that as
    /// `prefill_lse` at `window` merged with `pool_lse` and closed by `sink`.
    /// The window is fixed, so the only key set the indexer's budget can cap
    /// is the compressed one — which is also why the ratio-128 layers carry
    /// no indexer. `selection` is `[rows, top_k]` i32, ascending
    /// compressed-row ids with `-1` padding, and a row whose visible count
    /// fits its budget selects the identity and lands `attention_lse`'s own
    /// numbers.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_lse_selected(
        ctx: &Ctx<'_>,
        q: Tensor,
        positions: Tensor,
        request_of_token: Tensor,
        selection: Tensor,
        entries: &KvPool,
        ratio: u32,
        top_k: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Tensor,
        lse: Tensor,
    ) -> Result<(), Error> {
        const OP: &str = "attention.pool_lse_selected";
        dtype_dispatch!(OP, q.dtype, { Bf16 => () });
        debug_assert_eq!(lse.dtype, Dtype::F32, "`{OP}` lands an f32 log-sum-exp plane");
        debug_assert_eq!(
            selection.dtype,
            Dtype::I32,
            "`{OP}` walks the i32 compressed-row ids `attention.index_topk` published"
        );
        debug_assert_eq!(
            request_of_token.dtype, Dtype::I32,
            "`{OP}` reads an i32 owning request per token"
        );
        if entries.page_size <= 0 {
            return Err(refuse(OP, "the compressed cache page size is zero"));
        }
        let num_q_heads = nonzero(OP, "heads", heads)?;
        let head_dim = nonzero(OP, "the head width", head_dim)?;
        if head_dim > POOL_HEAD_MAX {
            return Err(refuse(
                OP,
                format!("the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in threadgroup memory"),
            ));
        }
        let rows = nonzero(OP, "rows", o.rows)?;
        let ratio = nonzero(OP, "the pooling ratio", ratio)?;
        let top_k = nonzero(OP, "the selection budget this reader states", top_k)?;
        if selection.width != top_k {
            return Err(refuse(
                OP,
                format!(
                    "the selection is {} wide and this reader walks the {top_k} ids it states",
                    selection.width
                ),
            ));
        }
        if selection.rows < rows {
            return Err(refuse(
                OP,
                format!(
                    "the selection seats {} rows and this reader launches {rows}",
                    selection.rows
                ),
            ));
        }
        ctx.fire(
            Fire::at(FILE, "pool_lse_selected_paged")
                .apply(Grid::of([ATTN_BLOCK, rows, num_q_heads], [ATTN_BLOCK, 1, 1])),
            &[
                q.arg(),
                entries.keys.arg(),
                selection.arg(),
                o.arg_mut(),
                lse.arg_mut(),
                positions.arg(),
                entries.page_indices.arg(),
                entries.page_indptr.arg(),
                request_of_token.arg(),
                stated(OP, num_q_heads)?.arg(),
                stated(OP, head_dim)?.arg(),
                stated(OP, ratio)?.arg(),
                stated(OP, top_k)?.arg(),
                entries.page_size.arg(),
                sm_scale.arg(),
            ],
        )
    }

    /// **THE SELECTED READER'S KEY WALK, IN HOST ARITHMETIC** — the deviceless
    /// pin on the one part of `pool_lse_selected_paged` that is a decision
    /// rather than a launch, in the idiom of
    /// [`index::bisect_select`](super::index::bisect_select).
    ///
    /// Given a row's selection and how many compressed rows it can see, this
    /// is the sequence of CELLS the shader reads, in the order it reads them:
    /// ids below zero are the pad and ids at or past the visible count are
    /// out of the causal bound, both SKIPPED rather than clamped, and a kept
    /// id `c` becomes the boundary cell `(c + 1) * ratio - 1` — the same
    /// arithmetic `pool_lse_paged` walks densely and `index_topk_paged` keys
    /// by.
    #[must_use]
    pub fn selected_cells(selection: &[i32], num_visible: i32, ratio: i32) -> Vec<i32> {
        selection
            .iter()
            .filter(|c| **c >= 0 && **c < num_visible)
            .map(|c| (c + 1) * ratio - 1)
            .collect()
    }

    /// **THE COMPRESSED ROW'S ROPE POSITION, IN HOST ARITHMETIC** — the
    /// deviceless twin of the boundary kernels' `out_rope` column, and the
    /// other half of [`selected_cells`]'s claim.
    ///
    /// A pooled entry has TWO positions and they are `ratio - 1` apart. The
    /// CELL it is cached at is the one its window closes on — `selected_cells`
    /// above, `(c + 1) · ratio - 1`, the block's LAST token — because that is
    /// the cell the readers address. The POSITION IT IS ROPED AT is the
    /// compressed row's own, `c · ratio`, the block's FIRST token: the
    /// reference ropes the pooled plane at `rows = arange(0, cutoff, ratio)`
    /// = `0, ratio, 2·ratio, …` (`v4mlx/compressor.py`'s `compressor_prefill`,
    /// both the attention compressor and the `rotate=True` indexer one), not
    /// at the tokens the block ended on.
    ///
    /// Given the closing token position `p` this returns that rope position.
    /// It is `(p / ratio) · ratio` and not `p`, which is the whole distance
    /// between the two readings.
    #[must_use]
    pub fn compressed_rope_pos(closing_pos: i32, ratio: i32) -> i32 {
        (closing_pos / ratio) * ratio
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::encode::ArgValue;
        use crate::probe::Probe;

        const HEAD_DIM: u32 = 512;
        const HEADS: u32 = 4;

        fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
            Tensor::new(buf, rows, width, Dtype::Bf16)
        }
        fn i32t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::I32)
        }
        fn u32t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::U32)
        }
        fn u8t(buf: u32, rows: u32) -> Tensor {
            Tensor::new(buf, rows, 1, Dtype::U8)
        }

        /// The compressed cache: `keys` are the pooled entry pages.
        fn comp_pool() -> KvPool {
            KvPool {
                keys: bf16(40, 4096, HEAD_DIM),
                values: bf16(41, 4096, HEAD_DIM),
                page_indices: u32t(42, 64),
                page_indptr: u32t(43, 8),
                page_size: 16,
                seq_stride: u64::from(HEAD_DIM),
                head_stride: u64::from(HEAD_DIM),
            }
        }

        /// The decode detector is one flat lane per token; its outputs are the
        /// THREE boundary tables — the cell, the lane and the compressed row's
        /// rope position — `n`/`ratio` follow, and `row_valid` is the mask.
        #[test]
        fn boundary_decode_is_flat_over_tokens_ratio_4() {
            let probe = Probe::default();
            boundary_decode(&probe, i32t(1, 6), u8t(2, 6), 4, i32t(3, 6), i32t(4, 6), i32t(5, 6))
                .expect("the decode detector enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.file, "attn/pool.metal");
            assert_eq!(f.entrypoint, "pool_boundary_decode");
            assert_eq!(f.lanes, [6, 1, 1]);
            assert_eq!(f.group, [META_BLOCK, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // positions
            assert_eq!(a[1], ArgValue::BufferMut(3)); // boundary_pos
            assert_eq!(a[2], ArgValue::BufferMut(4)); // boundary_req
            assert_eq!(a[3], ArgValue::BufferMut(5)); // boundary_rope
            assert_eq!(a[4], ArgValue::I32(6)); // n
            assert_eq!(a[5], ArgValue::I32(4)); // ratio
            assert_eq!(a[6], ArgValue::Buffer(2)); // row_valid
        }

        /// The ratio-128 detector selects the same point — the window arithmetic
        /// lives in the gather, not the boundary mark.
        #[test]
        fn boundary_decode_takes_ratio_128() {
            let probe = Probe::default();
            boundary_decode(&probe, i32t(1, 3), u8t(2, 3), 128, i32t(3, 3), i32t(4, 3), i32t(5, 3))
                .expect("ratio 128 enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "pool_boundary_decode");
            assert_eq!(a[5], ArgValue::I32(128));
        }

        /// **THE ROPE COLUMN IS ITS OWN PLANE.** The cell the entry is cached
        /// at and the position it is roped at are two different numbers for
        /// one entry (`(c+1)·ratio - 1` against `c·ratio`), so a shell that
        /// bound one buffer to both seats — the reading the model text had
        /// while this column did not exist — is the `ratio - 1` skew, and it
        /// is visible here without a device.
        #[test]
        fn the_rope_column_is_not_the_cell_column() {
            let probe = Probe::default();
            boundary_decode(&probe, i32t(1, 6), u8t(2, 6), 4, i32t(3, 6), i32t(4, 6), i32t(5, 6))
                .expect("the decode detector enqueues");
            let (_, a) = probe.only();
            assert_ne!(a[1], a[3], "the rope operand is not the cell operand");
        }

        /// The prefill detector adds the request indptr (buffer 1) and the
        /// request count for its binary search.
        #[test]
        fn boundary_prefill_carries_the_indptr_and_request_count() {
            let probe = Probe::default();
            let positions = RaggedTensor { data: i32t(1, 6), indptr: u32t(9, 3) };
            boundary_prefill(&probe, positions, u8t(2, 6), 4, i32t(3, 6), i32t(4, 6), i32t(5, 6))
                .expect("the prefill detector enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "pool_boundary_prefill");
            assert_eq!(f.lanes, [6, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // positions.data
            assert_eq!(a[1], ArgValue::Buffer(9)); // qo_indptr, not the data
            assert_eq!(a[2], ArgValue::BufferMut(3)); // boundary_pos
            assert_eq!(a[3], ArgValue::BufferMut(4)); // boundary_req
            assert_eq!(a[4], ArgValue::BufferMut(5)); // boundary_rope
            assert_eq!(a[5], ArgValue::I32(6)); // n
            assert_eq!(a[6], ArgValue::I32(2)); // num_requests = indptr.rows - 1
            assert_eq!(a[7], ArgValue::I32(4)); // ratio
            assert_eq!(a[8], ArgValue::Buffer(2)); // row_valid
        }

        /// The store is one threadgroup per entry over the head width; the
        /// compressed cache pages are the mutated plane, the boundary tables
        /// address the cell.
        #[test]
        fn kv_append_writes_the_compressed_pages_by_the_boundary_tables() {
            let probe = Probe::default();
            let pool = comp_pool();
            kv_append(&probe, bf16(1, 3, HEAD_DIM), i32t(5, 3), i32t(6, 3), &pool, u32t(7, 3), u32t(8, 3))
                .expect("the store enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "pool_store_entries_bfloat16");
            assert_eq!(f.lanes, [HEAD_DIM, 3, 1]);
            assert_eq!(f.group, [HEAD_DIM, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // entries
            assert_eq!(a[1], ArgValue::BufferMut(40)); // pool.keys
            assert_eq!(a[2], ArgValue::Buffer(5)); // boundary_pos
            assert_eq!(a[3], ArgValue::Buffer(6)); // boundary_req
            assert_eq!(a[4], ArgValue::Buffer(42)); // page_indices
            assert_eq!(a[5], ArgValue::Buffer(43)); // page_indptr
            assert_eq!(a[6], ArgValue::I32(HEAD_DIM as i32));
            assert_eq!(a[7], ArgValue::I32(16)); // page_size
        }

        /// The compressed-entry reader is one threadgroup per `(row, head)`,
        /// 128 threads wide; the pool is storage, `q`/`o`/`lse` the operands.
        #[test]
        fn attention_lse_is_a_threadgroup_per_row_head() {
            let probe = Probe::default();
            let pool = comp_pool();
            attention_lse(
                &probe, bf16(1, 2, HEADS * HEAD_DIM), i32t(2, 2), i32t(3, 2), &pool,
                4, HEADS, HEAD_DIM, 0.5, bf16(4, 2, HEADS * HEAD_DIM),
                Tensor::new(5, 2, HEADS, Dtype::F32),
            )
            .expect("the compressed reader enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "pool_lse_paged");
            assert_eq!(f.lanes, [ATTN_BLOCK, 2, HEADS]);
            assert_eq!(f.group, [ATTN_BLOCK, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // q
            assert_eq!(a[1], ArgValue::Buffer(40)); // entries.keys
            assert_eq!(a[2], ArgValue::BufferMut(4)); // o
            assert_eq!(a[3], ArgValue::BufferMut(5)); // lse
            assert_eq!(a[4], ArgValue::Buffer(2)); // positions
            assert_eq!(a[5], ArgValue::Buffer(42)); // page_indices
            assert_eq!(a[6], ArgValue::Buffer(43)); // page_indptr
            assert_eq!(a[7], ArgValue::Buffer(3)); // request_of_token
            assert_eq!(a[8], ArgValue::I32(HEADS as i32));
            assert_eq!(a[9], ArgValue::I32(HEAD_DIM as i32));
            assert_eq!(a[10], ArgValue::I32(4)); // ratio
            assert_eq!(a[11], ArgValue::I32(16)); // page_size
            assert_eq!(a[12], ArgValue::F32(0.5));
        }

        /// The SELECTED reader is the same launch shape with the selection
        /// bound between the pool and the outputs, and the budget stated
        /// beside the ratio the ids are compressed rows of.
        #[test]
        fn attention_lse_selected_binds_the_selection_and_its_budget() {
            let probe = Probe::default();
            let pool = comp_pool();
            attention_lse_selected(
                &probe, bf16(1, 2, HEADS * HEAD_DIM), i32t(2, 2), i32t(3, 2),
                Tensor::new(6, 2, 8, Dtype::I32), &pool, 4, 8, HEADS, HEAD_DIM,
                0.5, bf16(4, 2, HEADS * HEAD_DIM),
                Tensor::new(5, 2, HEADS, Dtype::F32),
            )
            .expect("the selected reader enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.entrypoint, "pool_lse_selected_paged");
            assert_eq!(f.lanes, [ATTN_BLOCK, 2, HEADS]);
            assert_eq!(f.group, [ATTN_BLOCK, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(1)); // q
            assert_eq!(a[1], ArgValue::Buffer(40)); // entries.keys
            assert_eq!(a[2], ArgValue::Buffer(6)); // selection
            assert_eq!(a[3], ArgValue::BufferMut(4)); // o
            assert_eq!(a[4], ArgValue::BufferMut(5)); // lse
            assert_eq!(a[5], ArgValue::Buffer(2)); // positions
            assert_eq!(a[6], ArgValue::Buffer(42)); // page_indices
            assert_eq!(a[7], ArgValue::Buffer(43)); // page_indptr
            assert_eq!(a[8], ArgValue::Buffer(3)); // request_of_token
            assert_eq!(a[9], ArgValue::I32(HEADS as i32));
            assert_eq!(a[10], ArgValue::I32(HEAD_DIM as i32));
            assert_eq!(a[11], ArgValue::I32(4)); // ratio
            assert_eq!(a[12], ArgValue::I32(8)); // top_k
            assert_eq!(a[13], ArgValue::I32(16)); // page_size
            assert_eq!(a[14], ArgValue::F32(0.5));
        }

        /// A selection whose width is not the budget the reader states is a
        /// row that would be walked off its own end. Refused, not launched.
        #[test]
        fn attention_lse_selected_refuses_a_selection_that_is_not_its_budget() {
            let probe = Probe::default();
            let pool = comp_pool();
            let why = attention_lse_selected(
                &probe, bf16(1, 2, HEADS * HEAD_DIM), i32t(2, 2), i32t(3, 2),
                Tensor::new(6, 2, 4, Dtype::I32), &pool, 4, 8, HEADS, HEAD_DIM,
                0.5, bf16(4, 2, HEADS * HEAD_DIM),
                Tensor::new(5, 2, HEADS, Dtype::F32),
            )
            .expect_err("a 4-wide selection is not an 8-id budget");
            assert!(format!("{why}").contains('8'), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// **THE SELECTED WALK REDUCES TO THE DENSE ONE.**
        /// `index_topk_paged`'s `nkeys <= topk` arm publishes the identity
        /// `0..nkeys-1` with `-1` padding, so a row inside its budget reads
        /// the same cells in the same order `pool_lse_paged` reads densely:
        /// `(c+1)*ratio - 1` for every visible `c`. That equality is what
        /// makes the fine branch safe to fire on short sequences, and it is
        /// checked here without a device.
        #[test]
        fn the_identity_selection_walks_the_dense_reader_s_own_cells() {
            let visible = 5;
            let ratio = 4;
            let identity = super::super::index::bisect_select(&[0.0; 5], 8);
            assert_eq!(identity, [0, 1, 2, 3, 4, -1, -1, -1]);
            let dense: Vec<i32> = (0..visible).map(|c| (c + 1) * ratio - 1).collect();
            assert_eq!(selected_cells(&identity, visible, ratio), dense);
            assert_eq!(dense, [3, 7, 11, 15, 19]);
        }

        /// The pad and the causal bound are both SKIPS and not clamps: a
        /// `-1` reads nothing, and an id at or past the query's own visible
        /// count reads nothing either. Order is the selection's.
        #[test]
        fn selected_cells_skips_the_pad_and_the_out_of_bound_id() {
            assert_eq!(selected_cells(&[-1, 0, 9, 2, -1], 5, 4), [3, 11]);
            assert_eq!(selected_cells(&[-1, -1, -1], 5, 4), Vec::<i32>::new());
            // `ratio == 1` is the per-token cache: the id IS the cell.
            assert_eq!(selected_cells(&[0, 3, 7], 8, 1), [0, 3, 7]);
        }

        /// **THE ROPE POSITIONS ARE THE REFERENCE'S `arange(0, cutoff, ratio)`.**
        /// Five closing boundaries at ratio 4 close on `3, 7, 11, 15, 19` — the
        /// cells above — and rope at `0, 4, 8, 12, 16`. That second row is the
        /// claim: it is the block STARTS, which is what `cos[rows]`/`sin[rows]`
        /// are indexed by in `compressor_prefill`.
        #[test]
        fn the_rope_positions_are_the_block_starts() {
            let ratio = 4;
            let closing: Vec<i32> = (0..5).map(|c| (c + 1) * ratio - 1).collect();
            let roped: Vec<i32> = closing
                .iter()
                .map(|p| compressed_rope_pos(*p, ratio))
                .collect();
            assert_eq!(closing, [3, 7, 11, 15, 19]);
            assert_eq!(roped, [0, 4, 8, 12, 16]);
            // `arange(0, cutoff, ratio)` said the other way, for `cutoff = 20`.
            let arange: Vec<i32> = (0..20).step_by(ratio as usize).collect();
            assert_eq!(roped, arange);
        }

        /// **AND THEY ARE NOT THE CLOSING POSITIONS.** The reading this
        /// replaced roped the pooled entry at the token position its window
        /// closed on, which is exactly `ratio - 1` too far on every entry —
        /// the delta this gate exists to score. At ratio 128 the same skew is
        /// 127 positions.
        #[test]
        fn the_old_reading_is_off_by_ratio_minus_one() {
            for ratio in [4, 128] {
                for c in 0..6 {
                    let closing = (c + 1) * ratio - 1;
                    let roped = compressed_rope_pos(closing, ratio);
                    assert_eq!(roped, c * ratio);
                    assert_eq!(closing - roped, ratio - 1);
                }
            }
            // `ratio == 1` is the degenerate compressor: one entry per token,
            // and the two positions coincide.
            assert_eq!(compressed_rope_pos(7, 1), 7);
        }

        /// A head wider than the flash reader's q tile is refused by name, not
        /// launched — the one geometry the shader's threadgroup array bounds.
        #[test]
        fn attention_lse_refuses_a_head_past_the_tile() {
            let probe = Probe::default();
            let pool = comp_pool();
            let why = attention_lse(
                &probe, bf16(1, 1, 640), i32t(2, 1), i32t(3, 1), &pool,
                4, 1, 640, 0.5, bf16(4, 1, 640), Tensor::new(5, 1, 1, Dtype::F32),
            )
            .expect_err("640 is past POOL_HEAD_MAX");
            assert!(format!("{why}").contains("512"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// The ratio-4 gather: `coff` is 2, so the state slabs are read at
        /// `2 x HEAD_DIM` and the ape plane is `[4, 2 x HEAD_DIM]` f32. The
        /// operand order is `pool.cuh`'s, with `has_ape` last — the Metal
        /// spelling of the CUDA `ape != nullptr`.
        #[test]
        fn gather_binds_the_three_slabs_at_coff_two() {
            let probe = Probe::default();
            let pool = comp_pool();
            let ape = Tensor::new(12, 4, 2 * HEAD_DIM, Dtype::F32);
            gather(
                &probe, i32t(1, 3), i32t(2, 3), &pool, HEAD_DIM, 4,
                bf16(10, 4096, 2 * HEAD_DIM), bf16(11, 4096, 2 * HEAD_DIM), Some(ape),
                bf16(3, 3, HEAD_DIM),
            )
            .expect("the gather enqueues");
            let (f, a) = probe.only();
            assert_eq!(f.file, "attn/pool.metal");
            assert_eq!(f.entrypoint, "pool_gather_paged_bfloat16");
            assert_eq!(f.lanes, [HEAD_DIM, 3, 1]);
            assert_eq!(f.group, [HEAD_DIM, 1, 1]);
            assert_eq!(a[0], ArgValue::Buffer(10)); // state_kv
            assert_eq!(a[1], ArgValue::Buffer(11)); // state_score
            assert_eq!(a[2], ArgValue::Buffer(12)); // ape
            assert_eq!(a[3], ArgValue::Buffer(1)); // boundary_pos
            assert_eq!(a[4], ArgValue::Buffer(2)); // boundary_req
            assert_eq!(a[5], ArgValue::Buffer(42)); // page_indices
            assert_eq!(a[6], ArgValue::Buffer(43)); // page_indptr
            assert_eq!(a[7], ArgValue::BufferMut(3)); // entries
            assert_eq!(a[8], ArgValue::I32(HEAD_DIM as i32));
            assert_eq!(a[9], ArgValue::I32(4)); // ratio
            assert_eq!(a[10], ArgValue::I32(2)); // coff
            assert_eq!(a[11], ArgValue::I32(16)); // page_size
            assert_eq!(a[12], ArgValue::I32(1)); // has_ape
        }

        /// The ratio-128 gather: `coff` is 1, the state slabs are the head
        /// width, and an absent ape plane states `has_ape = 0` while the seat
        /// takes `state_score`'s handle — Metal has no null buffer, and the
        /// shader never reads the seat it is told not to.
        #[test]
        fn gather_at_coff_one_states_no_ape_and_reuses_the_seat() {
            let probe = Probe::default();
            let pool = comp_pool();
            gather(
                &probe, i32t(1, 3), i32t(2, 3), &pool, HEAD_DIM, 128,
                bf16(10, 4096, HEAD_DIM), bf16(11, 4096, HEAD_DIM), None,
                bf16(3, 3, HEAD_DIM),
            )
            .expect("the ratio-128 gather enqueues");
            let (_, a) = probe.only();
            assert_eq!(a[1], ArgValue::Buffer(11)); // state_score
            assert_eq!(a[2], ArgValue::Buffer(11)); // the unread ape seat
            assert_eq!(a[9], ArgValue::I32(128)); // ratio
            assert_eq!(a[10], ArgValue::I32(1)); // coff
            assert_eq!(a[12], ArgValue::I32(0)); // has_ape
        }

        /// A state slab laid out at the wrong pitch is a launch reading
        /// somebody else's cells, so it is refused by name and nothing fires.
        #[test]
        fn gather_refuses_a_state_slab_of_the_wrong_pitch() {
            let probe = Probe::default();
            let pool = comp_pool();
            let why = gather(
                &probe, i32t(1, 3), i32t(2, 3), &pool, HEAD_DIM, 4,
                // `coff` is 2 at ratio 4, so a head-wide slab is half a row.
                bf16(10, 4096, HEAD_DIM), bf16(11, 4096, 2 * HEAD_DIM), None,
                bf16(3, 3, HEAD_DIM),
            )
            .expect_err("a half-pitch state_kv is refused");
            assert!(format!("{why}").contains("state_kv"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// And an ape plane whose rows are not the ratio is the same kind of
        /// mis-binding, caught on the plane the shader modulo-indexes.
        #[test]
        fn gather_refuses_an_ape_plane_that_is_not_a_block() {
            let probe = Probe::default();
            let pool = comp_pool();
            let why = gather(
                &probe, i32t(1, 3), i32t(2, 3), &pool, HEAD_DIM, 128,
                bf16(10, 4096, HEAD_DIM), bf16(11, 4096, HEAD_DIM),
                Some(Tensor::new(12, 4, HEAD_DIM, Dtype::F32)),
                bf16(3, 3, HEAD_DIM),
            )
            .expect_err("a 4-row ape plane is not a 128 block");
            assert!(format!("{why}").contains("ratio 128"), "{why}");
            assert!(probe.fires().is_empty());
        }

        /// `compressor_coff` is the overlapping-window fanout: `2` at ratio 4,
        /// `1` elsewhere — the shader's `coff`.
        #[test]
        fn coff_is_two_at_ratio_four_and_one_elsewhere() {
            assert_eq!(compressor_coff(4), 2);
            assert_eq!(compressor_coff(128), 1);
        }
    }
}
