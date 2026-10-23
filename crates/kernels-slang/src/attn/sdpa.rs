//! Paged attention over a KV cache (`attn.decode`, `attn.prefill` and their
//! LSE forms) from `kernels/attn/sdpa.slang`. Decode walks a row's keys with
//! the workgroup's subgroups on the subgroup tier or 32-lane strips on the
//! baseline; prefill tiles query rows of one head and walks their keys in
//! step.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Tier, instantiated, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const SDPA: &str = "attn/sdpa.slang";

/// The head widths the kernels are instantiated at.
const WIDTHS: [u32; 4] = [64, 128, 256, 512];

/// How an engine runs paged attention on its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tactic {
    /// Decode reduces with subgroup operations at `Tier::Subgroup` (SPIR-V
    /// only); prefill runs at the baseline whatever this says.
    pub tier: Tier,
    /// The device's subgroups run narrower than 32 lanes.
    pub narrow: bool,
    /// Invocations a decode workgroup walks with: 256, or 128.
    pub decode_threads: u32,
    /// Query rows a prefill workgroup holds: 32, 16 or 8.
    pub tile_rows: u32,
}

/// The per-token tables a fire's attention reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tables {
    pub positions: Tensor,
    pub request_of_token: Tensor,
    /// The packed byte mask, `mask_stride` bytes a row.
    pub mask: Tensor,
    /// One byte a row: whether the row reads its mask.
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
}

struct Shape {
    q_heads: u32,
    kv_heads: u32,
    rows: u32,
    window: i32,
}

fn shape(
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
) -> Result<Shape, Error> {
    if q.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op, dtype: q.dtype });
    }
    if !WIDTHS.contains(&head_dim) {
        return Err(refuse(
            op,
            format!("no sdpa kernel is instantiated at head width {head_dim}"),
        ));
    }
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    if pool.head_stride != u64::from(head_dim)
        || pool.seq_stride == 0
        || !pool.seq_stride.is_multiple_of(pool.head_stride)
    {
        return Err(refuse(
            op,
            format!(
                "the pool row's strides ({} a token, {} a head) are not whole {head_dim}-wide kv heads",
                pool.seq_stride, pool.head_stride
            ),
        ));
    }
    let kv_heads = u32::try_from(pool.seq_stride / pool.head_stride)
        .map_err(|_| refuse(op, "the pool's kv head count does not fit the shader's int"))?;
    let q_heads = super::row_heads(op, q.width, head_dim)?;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            op,
            format!(
                "{q_heads} query heads are not a whole number of the pool's {kv_heads} kv heads"
            ),
        ));
    }
    let window = match window {
        None => 0,
        Some(w) => stated(op, nonzero(op, "the sliding extent", w)?)?,
    };
    Ok(Shape {
        q_heads,
        kv_heads,
        rows: nonzero(op, "rows", q.rows)?,
        window,
    })
}

fn args(
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    tables: &Tables,
    shape: &Shape,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<Vec<ArgValue>, Error> {
    let mut args = vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        o.arg_mut(),
        tables.positions.arg(),
        tables.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        tables.mask.arg(),
        tables.mask_enabled.arg(),
    ];
    if let Some(lse) = lse {
        debug_assert_eq!(
            lse.dtype,
            Dtype::F32,
            "`{op}` lands an f32 log-sum-exp plane"
        );
        args.push(lse.arg_mut());
    }
    args.extend([
        stated(op, shape.q_heads / shape.kv_heads)?.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        tables.mask_stride.arg(),
        shape.window.arg(),
        stated(op, shape.q_heads)?.arg(),
    ]);
    Ok(args)
}

/// Each query row (one a token) attends its request's cached keys, causal,
/// one workgroup a (head, row); with `lse`, the row's base-2 LSE lands too.
pub fn decode(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    q: Tensor,
    tables: &Tables,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    const OP: &str = "attn.decode";
    let shape = shape(OP, q, pool, window, head_dim)?;
    let threads = if tactic.decode_threads >= 256 {
        256
    } else {
        128
    };
    let subgroup = tactic.tier == Tier::Subgroup;
    let name = format!(
        "attn_sdpa_{}{}{}_bf16_d_{head_dim}",
        if lse.is_some() {
            "decode_lse"
        } else {
            "decode"
        },
        if threads < 256 { "_t128" } else { "" },
        if subgroup && tactic.narrow {
            "_lanes_8"
        } else {
            ""
        },
    );
    let entry = instantiated(OP, name)?;
    let args = args(OP, q, pool, tables, &shape, sm_scale, o, lse)?;
    ctx.fire(
        Fire::at(SDPA, entry)
            .tier(if subgroup {
                Tier::Subgroup
            } else {
                Tier::Baseline
            })
            .groups([shape.q_heads, shape.rows, 1])
            .group([threads, 1, 1]),
        &args,
    )
}

/// Query rows tiled `tactic.tile_rows` a workgroup of one head, each attending
/// its request's cached keys, causal; with `lse`, each row's base-2 LSE.
pub fn prefill(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    q: Tensor,
    tables: &Tables,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    const OP: &str = "attn.prefill";
    let shape = shape(OP, q, pool, window, head_dim)?;
    let tile = match tactic.tile_rows {
        32.. => 32,
        16..=31 => 16,
        _ => 8,
    };
    let name = format!(
        "attn_sdpa_{}{}_bf16_d_{head_dim}",
        if lse.is_some() { "tiled_lse" } else { "tiled" },
        if tile == 32 {
            String::new()
        } else {
            format!("_r{tile}")
        },
    );
    let entry = instantiated(OP, name)?;
    let mut args = args(OP, q, pool, tables, &shape, sm_scale, o, lse)?;
    args.push(stated(OP, shape.rows)?.arg());
    ctx.fire(
        Fire::at(SDPA, entry)
            .groups([shape.q_heads, shape.rows.div_ceil(tile), 1])
            .group([32, tile, 1]),
        &args,
    )
}
