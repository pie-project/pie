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
//! [`mla`], [`index`], and [`pool`] answer typed refusals — the old plane
//! claimed none of their points either. [`ssm`], the recurrent mixer, sits
//! beside them. `gate` left for [`elemwise::gate`](crate::elemwise::gate):
//! it is elementwise, not attention.

pub mod ssm;

use kernels::KernelError;
use model_ir::Dtype;

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

const SDPA_LSE_WIDTHS: [u32; 1] = [64];

const SDPA_DECODE_LSE: [&str; 1] = ["sdpa_paged_decode_lse_bfloat16_d_64"];

const SDPA_TILED_LSE: [&str; 1] = ["sdpa_paged_tiled_lse_bfloat16_d_64"];

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
) -> Result<(), KernelError> {
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
) -> Result<DecodePlan, KernelError> {
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
) -> Result<PrefillPlan, KernelError> {
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

fn head_point(op: &'static str, head_dim: u32, points: &[u32]) -> Result<usize, KernelError> {
    points
        .iter()
        .position(|&p| p == head_dim)
        .ok_or_else(|| refuse(op, format!("no sdpa shader is stamped at head width {head_dim}")))
}

/// The sliding extent the shader reads: 0 is "no window", so a stated window
/// of zero is a degenerate statement, not an unwindowed one.
fn window_extent(op: &'static str, window: Option<u32>) -> Result<i32, KernelError> {
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
fn pool_heads(op: &'static str, pool: &KvPool, head_dim: u32) -> Result<u32, KernelError> {
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
) -> Result<(), KernelError> {
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

fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, KernelError> {
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
    ) -> Result<Self, KernelError> {
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

fn vector_grid(op: &'static str, q_heads: u32, rows: u32) -> Result<[u32; 3], KernelError> {
    let x = q_heads.checked_mul(SDPA_THREADS).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {q_heads} query heads, one {SDPA_THREADS}-thread group each"),
        )
    })?;
    Ok([x, rows, 1])
}

fn tiled_grid(op: &'static str, q_heads: u32, rows: u32) -> Result<[u32; 3], KernelError> {
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
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), KernelError> {
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
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), KernelError> {
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
    pool: &KvPool,
    plan: &PrefillPlan,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), KernelError> {
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
    pool: &KvPool,
    plan: &PrefillPlan,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), KernelError> {
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
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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

/// The metal plane never claimed this point; the refusal is typed now.
#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    _ctx: &Ctx<'_>,
    _o1: Tensor,
    _lse1: Tensor,
    _o2: Tensor,
    _lse2: Tensor,
    _heads: u32,
    _head_dim: u32,
    _o: Tensor,
    _lse: Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported {
        op: "attention.merge_lse",
    })
}

/// `x = cap * tanh(x / cap)`, in place on `x`.
pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), KernelError> {
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
fn head_split(op: &'static str, pool: &KvPool, row: u32) -> Result<(u32, u32), KernelError> {
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
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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

/// `Mla`: multi-head latent attention. The metal plane never claimed any of
/// these points — the old file held an empty claims impl — so every entry is
/// a typed refusal and the driver arm stays destructure → resolve → call.
pub mod mla {
    use kernels::KernelError;

    use crate::encode::Ctx;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    /// Never constructed: `mla.plan` refuses before one exists. The type
    /// keeps the driver's plan arm shaped like the claimed families'.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MlaPlan;

    pub fn plan(
        _ctx: &Ctx<'_>,
        _kv_indptr: Tensor,
        _kv_indices: Tensor,
        _last_page_len: Tensor,
        _kv_len: Tensor,
    ) -> Result<MlaPlan, KernelError> {
        Err(KernelError::Unsupported { op: "mla.plan" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn latents(
        _ctx: &Ctx<'_>,
        _kv_a: Tensor,
        _weight: Tensor,
        _eps: f32,
        _kv_lora_rank: u32,
        _kv_c: Tensor,
        _k_pe: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "mla.latents" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn latents_rope(
        _ctx: &Ctx<'_>,
        _kv_a: Tensor,
        _positions: Tensor,
        _weight: Tensor,
        _eps: f32,
        _kv_lora_rank: u32,
        _rope_dim: u32,
        _theta: f32,
        _kv_c: Tensor,
        _k_pe: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.latents_rope",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_q_b(
        _ctx: &Ctx<'_>,
        _q_b: Tensor,
        _heads: u32,
        _nope_dim: u32,
        _rope_dim: u32,
        _q_nope: Tensor,
        _q_pe: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.split_q_b",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn absorb_q(
        _ctx: &Ctx<'_>,
        _q_nope: Tensor,
        _kv_b: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _nope_dim: u32,
        _v_head_dim: u32,
        _q_latent: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "mla.absorb_q" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn absorb_out(
        _ctx: &Ctx<'_>,
        _latent: Tensor,
        _kv_b: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _v_head_dim: u32,
        _nope_dim: u32,
        _o: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.absorb_out",
        })
    }

    pub fn kv_append(
        _ctx: &Ctx<'_>,
        _kv_c: Tensor,
        _k_pe: Tensor,
        _pool: &KvPool,
        _write_page: Tensor,
        _write_offset: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.kv_append",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode(
        _ctx: &Ctx<'_>,
        _q: Tensor,
        _pool: &KvPool,
        _plan: &MlaPlan,
        _q_pe: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _sm_scale: f32,
        _o: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.attention_decode",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill(
        _ctx: &Ctx<'_>,
        _q: RaggedTensor,
        _pool: &KvPool,
        _plan: &MlaPlan,
        _q_pe: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _sm_scale: f32,
        _o: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.attention_prefill",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_selected(
        _ctx: &Ctx<'_>,
        _q: Tensor,
        _pool: &KvPool,
        _plan: &MlaPlan,
        _q_pe: Tensor,
        _selection: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _sm_scale: f32,
        _o: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.attention_decode_selected",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_selected(
        _ctx: &Ctx<'_>,
        _q: RaggedTensor,
        _pool: &KvPool,
        _plan: &MlaPlan,
        _q_pe: Tensor,
        _selection: Tensor,
        _heads: u32,
        _kv_lora_rank: u32,
        _sm_scale: f32,
        _o: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "mla.attention_prefill_selected",
        })
    }
}

/// `Index`: the sparse-attention indexer. Unclaimed on metal, as before —
/// typed refusals only.
pub mod index {
    use kernels::KernelError;

    use crate::encode::Ctx;
    use crate::tensor::{KvPool, Tensor};

    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_rope(
        _ctx: &Ctx<'_>,
        _k: Tensor,
        _positions: Tensor,
        _weight: Tensor,
        _bias: Tensor,
        _eps: f32,
        _rope_dim: u32,
        _theta: f32,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "index.layernorm_rope",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        _ctx: &Ctx<'_>,
        _q: Tensor,
        _positions: Tensor,
        _heads: u32,
        _head_dim: u32,
        _rope_dim: u32,
        _theta: f32,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "index.rope" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn topk(
        _ctx: &Ctx<'_>,
        _q: Tensor,
        _weights: Tensor,
        _keys: &KvPool,
        _heads: u32,
        _head_dim: u32,
        _top_k: u32,
        _selection: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "index.topk" })
    }

    pub fn kv_append(
        _ctx: &Ctx<'_>,
        _k: Tensor,
        _keys: &KvPool,
        _write_page: Tensor,
        _write_offset: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "index.kv_append",
        })
    }
}

/// `Pool`: pooled (compressed) attention. Unclaimed on metal, as before —
/// typed refusals only.
pub mod pool {
    use kernels::KernelError;

    use crate::encode::Ctx;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    pub fn boundary_decode(
        _ctx: &Ctx<'_>,
        _positions: Tensor,
        _row_valid: Tensor,
        _ratio: u32,
        _boundary_pos: Tensor,
        _boundary_req: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "pool.boundary_decode",
        })
    }

    pub fn boundary_prefill(
        _ctx: &Ctx<'_>,
        _positions: RaggedTensor,
        _row_valid: Tensor,
        _ratio: u32,
        _boundary_pos: Tensor,
        _boundary_req: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "pool.boundary_prefill",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather(
        _ctx: &Ctx<'_>,
        _boundary_pos: Tensor,
        _boundary_req: Tensor,
        _pages: &KvPool,
        _head_dim: u32,
        _ratio: u32,
        _entries: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported { op: "pool.gather" })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_append(
        _ctx: &Ctx<'_>,
        _entries: Tensor,
        _boundary_pos: Tensor,
        _boundary_req: Tensor,
        _pool: &KvPool,
        _write_page: Tensor,
        _write_offset: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "pool.kv_append",
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_lse(
        _ctx: &Ctx<'_>,
        _q: Tensor,
        _positions: Tensor,
        _request_of_token: Tensor,
        _entries: &KvPool,
        _ratio: u32,
        _heads: u32,
        _head_dim: u32,
        _sm_scale: f32,
        _o: Tensor,
        _lse: Tensor,
    ) -> Result<(), KernelError> {
        Err(KernelError::Unsupported {
            op: "pool.attention_lse",
        })
    }
}
