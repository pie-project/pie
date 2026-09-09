pub mod arbiter;

pub mod dense;

pub mod merge;

pub mod ple;

pub mod score;

pub mod ssm;

pub mod dynconv;

pub mod selector;

pub mod ragged;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub positions: Tensor,

    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub positions: Tensor,

    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

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
    points.iter().position(|&p| p == head_dim).ok_or_else(|| {
        refuse(
            op,
            format!("no sdpa shader is stamped at head width {head_dim}"),
        )
    })
}

fn window_extent(op: &'static str, window: Option<u32>) -> Result<i32, Error> {
    match window {
        None => Ok(0),
        Some(w) => {
            nonzero(op, "the sliding extent this attention states", w)?;
            stated(op, w)
        }
    }
}

fn encoded_window(op: &'static str, window: Option<u32>, causal: bool) -> Result<i32, Error> {
    let extent = window_extent(op, window)?;
    Ok(if causal { extent } else { -(extent + 1) })
}

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
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the {width}-wide query row does not divide by the stated head width {head_dim}"
            ),
        ));
    }
    Ok(width / head_dim)
}

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
        causal: bool,
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
            window: encoded_window(op, window, causal)?,
            rows: nonzero(op, "rows", q.rows)?,
            at: head_point(op, head_dim, &SDPA_WIDTHS)?,
        })
    }
}

fn lse_plane(op: &'static str, lse: Tensor, shape: &Paged) {
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{op}` lands an f32 log-sum-exp plane"
    );
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

#[allow(clippy::too_many_arguments)]
fn vector(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    causal: bool,
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
    let shape = Paged::of(op, q, pool, window, causal, head_dim)?;
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
        ctx.absent()?,
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

#[allow(clippy::too_many_arguments)]
fn tiled(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    causal: bool,
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
    let shape = Paged::of(op, q, pool, window, causal, head_dim)?;
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
        ctx.absent()?,
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
        true,
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
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
}

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
        ctx, OP, q.data, pool, plan, plan.mask, window, true, head_dim, sm_scale, o, None,
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
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    causal: bool,
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
        ctx, OP, q.data, pool, plan, mask, window, causal, head_dim, sm_scale, o, None,
    )
}

pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    let entry = dtype_dispatch!(OP, o.dtype, { Bf16 => "attn_sink_rescale_bfloat16" });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 log-sum-exp plane"
    );
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

pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attention.logit_softcap";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "logit_softcap_bfloat16" });
    ctx.fire(
        Fire::at("attn/logit_softcap.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [256, 1, 1])),
        &[x.arg(), x.arg_mut(), cap.arg()],
    )
}

fn head_split(op: &'static str, pool: &KvPool, row: u32) -> Result<(u32, u32), Error> {
    let head_dim = u32::try_from(pool.head_stride)
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the pool row's head stride {} spells no head width",
                    pool.head_stride
                ),
            )
        })?;
    if row == 0 || !row.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the {row}-wide appended row does not divide by the pool's head stride {head_dim}"
            ),
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
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            ctx.absent()?,
            pool.page_size.arg(),
            ctx.absent()?,
            stated(op, heads)?.arg(),
            write_page.arg(),
            write_offset.arg(),
            0_i32.arg(),
        ],
    )
}

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

pub mod mla {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/mla.metal";

    const PREP_THREADS: u32 = 256;

    const SIMD: u32 = 32;

    const MAX_CKV: u32 = 16 * SIMD;
    const MAX_KPE: u32 = 4 * SIMD;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MlaPlan;

    pub fn plan(
        _ctx: &Ctx<'_>,
        _kv_indptr: Tensor,
        _kv_indices: Tensor,
        _last_page_len: Tensor,
        _kv_len: Tensor,
    ) -> Result<MlaPlan, Error> {
        Ok(MlaPlan)
    }

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
        split_kv_a_norm(
            ctx,
            "attention.mla_latents",
            kv_a,
            weight,
            eps,
            kv_lora_rank,
            kv_c,
            k_pe,
        )
    }

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
            Fire::at(FILE, entry)
                .apply(Grid::of([PREP_THREADS * rows, 1, 1], [PREP_THREADS, 1, 1])),
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
        let total = i32::try_from(total).map_err(|_| {
            refuse(
                OP,
                format!("{total} split elements do not fit the shader's int"),
            )
        })?;
        let lanes =
            u32::try_from(total).map_err(|_| refuse(OP, "the split grid will not launch"))?;
        debug_assert!(
            q_nope.width == u32::try_from(i64::from(heads) * i64::from(nope)).unwrap_or(u32::MAX)
                && q_pe.width
                    == u32::try_from(i64::from(heads) * i64::from(rope)).unwrap_or(u32::MAX),
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
            Fire::at(FILE, entry).apply(Grid::of(
                [kv_lora_rank, heads, rows],
                [SIMD.min(kv_lora_rank), 1, 1],
            )),
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
        let v_dim = stated(
            OP,
            crate::encode::nonzero(OP, "the value head dim", v_head_dim)?,
        )?;
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
            Fire::at(FILE, entry).apply(Grid::of(
                [v_head_dim, heads, rows],
                [SIMD.min(v_head_dim), 1, 1],
            )),
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
        if !q_pe.width.is_multiple_of(heads) {
            return Err(refuse(
                op,
                format!(
                    "the {}-wide rotated q plane does not divide by the {heads} heads",
                    q_pe.width
                ),
            ));
        }
        let kpe = if q_pe.width == 0 {
            0
        } else {
            latent_strip(op, "rope width", q_pe.width / heads, MAX_KPE)?
        };
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
            Fire::at(FILE, entry).apply(Grid::of(
                [heads * SIMD * MLA_SPLIT, rows, 1],
                [SIMD * MLA_SPLIT, 1, 1],
            )),
            &args,
        )
    }

    const MLA_SPLIT: u32 = 8;

    fn latent_strip(op: &'static str, what: &str, width: u32, max: u32) -> Result<i32, Error> {
        if width == 0 || !width.is_multiple_of(SIMD) || width > max {
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

        #[test]
        fn attn_every_case() {
            a_rank_the_strips_cannot_hold_is_refused();
            a_zero_value_width_is_refused_rather_than_launched();
            a_selection_that_is_not_one_row_per_query_is_refused();
            the_selected_point_refuses_the_geometries_the_dense_one_does();
        }

        fn a_rank_the_strips_cannot_hold_is_refused() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode(
                &probe,
                bf16(1, 1, HEADS * 500),
                bf16(2, 1, HEADS * ROPE),
                &pool,
                i32t(7, 1),
                i32t(8, 1),
                HEADS,
                500,
                0.5,
                bf16(3, 1, HEADS * 500),
            )
            .expect_err("500 is not a multiple of 32");
            assert!(format!("{why}").contains("lane-split"), "{why}");
            assert!(probe.fires().is_empty());
        }

        fn a_zero_value_width_is_refused_rather_than_launched() {
            let probe = Probe::default();
            let why = absorb_out(
                &probe,
                bf16(1, 1, HEADS * RANK),
                bf16(2, HEADS * NOPE, RANK),
                HEADS,
                RANK,
                0,
                NOPE,
                bf16(3, 1, 1),
            )
            .expect_err("a zero value head dim");
            assert!(format!("{why}").contains("value head dim"), "{why}");
            assert!(probe.fires().is_empty());
        }

        fn sel(buf: u32, rows: u32, top_k: u32) -> Tensor {
            Tensor::new(buf, rows, top_k, Dtype::I32)
        }

        fn a_selection_that_is_not_one_row_per_query_is_refused() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode_selected(
                &probe,
                bf16(1, 2, HEADS * RANK),
                bf16(2, 2, HEADS * ROPE),
                sel(4, 1, 128),
                &pool,
                i32t(7, 2),
                i32t(8, 2),
                HEADS,
                RANK,
                0.5,
                bf16(3, 2, HEADS * RANK),
            )
            .expect_err("one selection row does not serve two query rows");
            assert!(format!("{why}").contains("one row per query row"), "{why}");
            assert!(probe.fires().is_empty());
        }

        fn the_selected_point_refuses_the_geometries_the_dense_one_does() {
            let probe = Probe::default();
            let pool = latent_pool();
            let why = attention_decode_selected(
                &probe,
                bf16(1, 1, HEADS * 500),
                bf16(2, 1, HEADS * ROPE),
                sel(4, 1, 8),
                &pool,
                i32t(7, 1),
                i32t(8, 1),
                HEADS,
                500,
                0.5,
                bf16(3, 1, HEADS * 500),
            )
            .expect_err("500 is not a multiple of 32");
            assert!(format!("{why}").contains("lane-split"), "{why}");
            assert!(probe.fires().is_empty());
        }
    }
}

pub mod index {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, Tensor};

    const FILE: &str = "attn/index.metal";

    const K_BLOCK: u32 = 256;

    const MAX_ROPE_DIM: u32 = 256;

    #[must_use]
    fn q_rope_block(n_heads: u32) -> u32 {
        (n_heads.div_ceil(32) * 32).max(32)
    }

    fn rotated(op: &'static str, rope_dim: u32, head_dim: u32) -> Result<i32, Error> {
        if !rope_dim.is_multiple_of(2) {
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
        let no_rope = Tensor::new(k.buf, k.rows, 0, k.dtype);
        super::mla::kv_append(ctx, k, no_rope, keys, write_page, write_offset)
    }

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
        debug_assert_eq!(
            selection.dtype,
            Dtype::I32,
            "`{OP}` writes i32 cached positions"
        );
        debug_assert_eq!(scores.dtype, Dtype::F32, "`{OP}` bisects an f32 score slab");
        debug_assert!(
            positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
            "the fire's position and owning-request tables are i32, one entry per row"
        );
        let heads = nonzero(OP, "the head count this ranking states", heads)?;
        let head_dim = nonzero(OP, "the key width this ranking states", head_dim)?;
        let top_k = nonzero(OP, "the selection budget this ranking states", top_k)?;
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

        use crate::probe::Probe;

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

        #[test]
        fn attn_1_every_case() {
            kv_append_refuses_a_pool_that_is_not_one_row_per_token();
            topk_refuses_a_zero_key_stride();
            topk_refuses_the_three_shapes_its_cuda_twin_refuses();
            topk_refuses_a_score_slab_shorter_than_the_launch();
        }

        fn kv_append_refuses_a_pool_that_is_not_one_row_per_token() {
            let probe = Probe::default();
            let pool = KvPool {
                seq_stride: u64::from(DIM) * 2,
                ..index_pool()
            };
            let why = kv_append(&probe, bf16(1, 4, DIM), &pool, u32t(7, 4), u32t(8, 4))
                .expect_err("a doubled pitch is not this row");
            assert!(format!("{why}").contains("token pitch"), "{why}");
            assert!(probe.fires().is_empty());
        }

        fn topk_refuses_a_zero_key_stride() {
            let probe = Probe::default();
            let pool = index_pool();
            let why = topk(
                &probe,
                bf16(1, 1, HEADS * DIM),
                bf16(2, 1, HEADS),
                &pool,
                i32t(9, 1, 1),
                i32t(10, 1, 1),
                f32t(11, 1, 8192),
                HEADS,
                DIM,
                TOPK,
                0,
                i32t(12, 1, TOPK),
            )
            .expect_err("zero is not a key stride");
            assert!(format!("{why}").contains("key stride"), "{why}");
            assert!(probe.fires().is_empty());
        }

        fn topk_refuses_the_three_shapes_its_cuda_twin_refuses() {
            let probe = Probe::default();
            let pool = index_pool();
            let bad_q = topk(
                &probe,
                bf16(1, 1, HEADS * DIM + 8),
                bf16(2, 1, HEADS),
                &pool,
                i32t(9, 1, 1),
                i32t(10, 1, 1),
                f32t(11, 1, 8192),
                HEADS,
                DIM,
                TOPK,
                1,
                i32t(12, 1, TOPK),
            )
            .expect_err("the query does not divide");
            assert!(format!("{bad_q}").contains("does not divide"), "{bad_q}");

            let bad_w = topk(
                &probe,
                bf16(1, 1, HEADS * DIM),
                bf16(2, 1, HEADS - 1),
                &pool,
                i32t(9, 1, 1),
                i32t(10, 1, 1),
                f32t(11, 1, 8192),
                HEADS,
                DIM,
                TOPK,
                1,
                i32t(12, 1, TOPK),
            )
            .expect_err("the weights are not one per head");
            assert!(
                format!("{bad_w}").contains("one per stated head"),
                "{bad_w}"
            );

            let bad_sel = topk(
                &probe,
                bf16(1, 1, HEADS * DIM),
                bf16(2, 1, HEADS),
                &pool,
                i32t(9, 1, 1),
                i32t(10, 1, 1),
                f32t(11, 1, 8192),
                HEADS,
                DIM,
                TOPK,
                1,
                i32t(12, 1, TOPK - 1),
            )
            .expect_err("the selection is not the budget");
            assert!(
                format!("{bad_sel}").contains("budget it stated"),
                "{bad_sel}"
            );

            assert!(probe.fires().is_empty());
        }

        fn topk_refuses_a_score_slab_shorter_than_the_launch() {
            let probe = Probe::default();
            let pool = index_pool();
            let why = topk(
                &probe,
                bf16(1, 8, HEADS * DIM),
                bf16(2, 8, HEADS),
                &pool,
                i32t(9, 8, 1),
                i32t(10, 8, 1),
                f32t(11, 4, 8192),
                HEADS,
                DIM,
                TOPK,
                1,
                i32t(12, 8, TOPK),
            )
            .expect_err("four slab rows do not seat eight query rows");
            assert!(format!("{why}").contains("seats 4 rows"), "{why}");
            assert!(probe.fires().is_empty());
        }
    }
}

pub mod pool {
    use dtype::Dtype;

    use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
    use crate::error::Error;
    use crate::tensor::{KvPool, RaggedTensor, Tensor};

    const FILE: &str = "attn/pool.metal";

    const META_BLOCK: u32 = 128;

    const ATTN_BLOCK: u32 = 128;

    const POOL_HEAD_MAX: u32 = 512;

    const fn compressor_coff(ratio: u32) -> i32 {
        if ratio == 4 { 2 } else { 1 }
    }

    fn boundary_tables(op: &'static str, boundary_pos: &Tensor, boundary_req: &Tensor) {
        debug_assert_eq!(
            boundary_pos.dtype,
            Dtype::I32,
            "`{op}` reads i32 boundary positions"
        );
        debug_assert_eq!(
            boundary_req.dtype,
            Dtype::I32,
            "`{op}` reads i32 boundary requests"
        );
        debug_assert_eq!(
            boundary_pos.rows, boundary_req.rows,
            "`{op}`'s boundary tables are one entry per token row"
        );
    }

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
        let width = nonzero(OP, "the compressor's row width", kv.width)?;
        if (width != head_dim && width != 2 * head_dim) || score.width != width {
            return Err(refuse(
                OP,
                format!(
                    "a ratio-{ratio} compressor projects head width {head_dim} or twice \
                     it; the pair handed over is {} and {}",
                    kv.width, score.width
                ),
            ));
        }
        let coff = width / head_dim;
        debug_assert_eq!(
            score.rows, kv.rows,
            "the two projections are one row per token row"
        );
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
        let coff = match ape {
            None => compressor_coff(ratio),
            Some(ape) if ape.width == head_dim => 1,
            Some(ape) if ape.width == 2 * head_dim => 2,
            Some(ape) => {
                return Err(refuse(
                    OP,
                    format!(
                        "an ape {} wide is neither one head width ({head_dim}) nor two",
                        ape.width
                    ),
                ));
            }
        };
        let width = head_dim.saturating_mul(coff.unsigned_abs());
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
                format!(
                    "the pooled entry width {head_dim} is above the {POOL_HEAD_MAX} this store launches as one threadgroup"
                ),
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
        debug_assert_eq!(
            lse.dtype,
            Dtype::F32,
            "`{OP}` lands an f32 log-sum-exp plane"
        );
        debug_assert_eq!(
            request_of_token.dtype,
            Dtype::I32,
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
                format!(
                    "the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in threadgroup memory"
                ),
            ));
        }
        let rows = nonzero(OP, "rows", o.rows)?;
        let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
        ctx.fire(
            Fire::at(FILE, "pool_lse_paged").apply(Grid::of(
                [ATTN_BLOCK, rows, num_q_heads],
                [ATTN_BLOCK, 1, 1],
            )),
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
        debug_assert_eq!(
            lse.dtype,
            Dtype::F32,
            "`{OP}` lands an f32 log-sum-exp plane"
        );
        debug_assert_eq!(
            selection.dtype,
            Dtype::I32,
            "`{OP}` walks the i32 compressed-row ids `attention.index_topk` published"
        );
        debug_assert_eq!(
            request_of_token.dtype,
            Dtype::I32,
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
                format!(
                    "the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in threadgroup memory"
                ),
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
            Fire::at(FILE, "pool_lse_selected_paged").apply(Grid::of(
                [ATTN_BLOCK, rows, num_q_heads],
                [ATTN_BLOCK, 1, 1],
            )),
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

    #[must_use]
    pub fn selected_cells(selection: &[i32], num_visible: i32, ratio: i32) -> Vec<i32> {
        selection
            .iter()
            .filter(|c| **c >= 0 && **c < num_visible)
            .map(|c| (c + 1) * ratio - 1)
            .collect()
    }

    #[must_use]
    pub fn compressed_rope_pos(closing_pos: i32, ratio: i32) -> i32 {
        (closing_pos / ratio) * ratio
    }
}
