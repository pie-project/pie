//! Paged attention over a KV cache (`attn.decode`, `attn.prefill`, their
//! LSE forms, and the selected forms) from `kernels/attn/sdpa.slang`.
//! Decode walks a row's keys with the workgroup's subgroups on the subgroup
//! tier or 32-lane strips on the baseline, a long row split into key ranges
//! folded by their LSEs where the device would otherwise idle; prefill tiles
//! query rows of one head and walks their keys in step.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, Tier, instantiated, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const SDPA: &str = "attn/sdpa.slang";

/// The head widths the kernels are instantiated at.
const WIDTHS: [u32; 4] = [64, 128, 256, 512];

/// The workgroup width the split fold is built at.
const FOLD_GROUP: u32 = 256;

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
    /// Decode workgroups the device wants in flight: a decode with fewer
    /// (heads x rows) splits its rows' keys to reach them.
    pub split_groups: u32,
    /// The fewest keys a split walks.
    pub split_min_keys: u32,
}

/// The f32 planes a split decode lands its partials in: `part_out` a head
/// row a (split, row, head), `part_lse` one value each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    pub part_out: Tensor,
    pub part_lse: Tensor,
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

impl Tables {
    /// `attn.plan_decode` and `attn.plan_prefill`: the fire's tables, refused
    /// where a kernel would read them as something else.
    pub fn plan(op: &'static str, tables: Tables) -> Result<Tables, Error> {
        let stated = [
            ("position table", tables.positions, Dtype::I32),
            ("owning-request table", tables.request_of_token, Dtype::I32),
            ("mask planes", tables.mask, Dtype::U8),
            ("mask-enabled flags", tables.mask_enabled, Dtype::U8),
        ];
        for (what, t, want) in stated {
            if t.dtype != want {
                return Err(refuse(
                    op,
                    format!(
                        "the fire's {what} is {:?}, and a plan carries {want:?}",
                        t.dtype
                    ),
                ));
            }
        }
        if tables.positions.rows != tables.request_of_token.rows {
            return Err(refuse(
                op,
                format!(
                    "the fire tables disagree: {} positions beside {} owning requests, and \
                     both are one entry per token",
                    tables.positions.rows, tables.request_of_token.rows
                ),
            ));
        }
        Ok(tables)
    }
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

/// The buffers, then `extra` (bound after the tables), then the scalars
/// every form reads.
fn args(
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    tables: &Tables,
    shape: &Shape,
    sm_scale: f32,
    o: Tensor,
    extra: &[ArgValue],
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
    args.extend_from_slice(extra);
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

fn lse_plane(op: &'static str, lse: Tensor, shape: &Shape) -> Result<ArgValue, Error> {
    let held = u64::from(lse.rows) * u64::from(lse.width);
    if lse.dtype != Dtype::F32 || held < u64::from(shape.rows) * u64::from(shape.q_heads) {
        return Err(refuse(
            op,
            format!(
                "the log-sum-exp plane is {} x {} {:?}, and it holds an f32 a head a row ({} x {})",
                lse.rows, lse.width, lse.dtype, shape.rows, shape.q_heads
            ),
        ));
    }
    Ok(lse.arg_mut())
}

/// A decode workgroup's entry: `kind`, then its thread and lane variants.
fn decode_entry(
    op: &'static str,
    tactic: Tactic,
    kind: &str,
    head_dim: u32,
) -> Result<Fire, Error> {
    let threads = if tactic.decode_threads >= 256 {
        256
    } else {
        128
    };
    let subgroup = tactic.tier == Tier::Subgroup;
    let name = format!(
        "attn_sdpa_{kind}{}{}_bf16_d_{head_dim}",
        if threads < 256 { "_t128" } else { "" },
        if subgroup && tactic.narrow {
            "_lanes_8"
        } else {
            ""
        },
    );
    let tier = if subgroup {
        Tier::Subgroup
    } else {
        Tier::Baseline
    };
    Ok(Fire::at(SDPA, instantiated(op, name)?)
        .tier(tier)
        .group([threads, 1, 1]))
}

/// Each query row (one a token) attends its request's cached keys, causal,
/// one workgroup a (head, row); with `lse`, the row's base-2 LSE lands too.
/// With `split`, rows too few to fill the device split their keys into
/// ranges, each landing a partial the fold then weighs by its LSE.
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
    split: Option<Split>,
) -> Result<(), Error> {
    const OP: &str = "attn.decode";
    let shape = shape(OP, q, pool, window, head_dim)?;
    if let Some(planes) = split {
        let ceiling = split_ceiling(&planes, &shape, head_dim);
        // Every page the fire's tables name: the walker splits no row past
        // the keys it holds.
        let keys = pool
            .page_indices
            .rows
            .saturating_mul(pool.page_indices.width)
            .saturating_mul(u32::try_from(pool.page_size).unwrap_or(0));
        let slices = splits_for(&tactic, &shape, ceiling, window, keys);
        if slices > 1 {
            return decode_split(
                ctx, tactic, q, tables, pool, &shape, head_dim, sm_scale, o, lse, planes, slices,
            );
        }
    }
    let mut extra = Vec::new();
    if let Some(lse) = lse {
        extra.push(lse_plane(OP, lse, &shape)?);
    }
    let fire = decode_entry(
        OP,
        tactic,
        if lse.is_some() {
            "decode_lse"
        } else {
            "decode"
        },
        head_dim,
    )?;
    let args = args(OP, q, pool, tables, &shape, sm_scale, o, &extra)?;
    ctx.fire(fire.groups([shape.q_heads, shape.rows, 1]), &args)
}

/// The splits a decode of this shape takes: enough (heads x rows x splits)
/// workgroups to reach the device's want, no split under its fewest keys,
/// and no more than the partial planes hold.
fn splits_for(tactic: &Tactic, shape: &Shape, ceiling: u32, window: Option<u32>, keys: u32) -> u32 {
    let groups = shape.q_heads.saturating_mul(shape.rows).max(1);
    if ceiling <= 1 || groups >= tactic.split_groups {
        return 1;
    }
    let min_keys = tactic.split_min_keys.max(1);
    let bound = window.map_or(keys, |w| w.min(keys));
    tactic
        .split_groups
        .div_ceil(groups)
        .min(bound.div_ceil(min_keys))
        .clamp(1, ceiling.max(1))
}

/// The splits the partial planes hold for this shape.
fn split_ceiling(split: &Split, shape: &Shape, head_dim: u32) -> u32 {
    let out_room = u64::from(split.part_out.rows) * u64::from(split.part_out.width);
    let lse_room = u64::from(split.part_lse.rows) * u64::from(split.part_lse.width);
    let lse_slice = u64::from(shape.rows) * u64::from(shape.q_heads);
    let out_slice = lse_slice * u64::from(head_dim);
    if out_slice == 0 {
        return 0;
    }
    u32::try_from((out_room / out_slice).min(lse_room / lse_slice)).unwrap_or(u32::MAX)
}

fn decode_split(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    q: Tensor,
    tables: &Tables,
    pool: &KvPool,
    shape: &Shape,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    split: Split,
    slices: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.decode_split";
    if split.part_out.dtype != Dtype::F32 || split.part_lse.dtype != Dtype::F32 {
        return Err(refuse(OP, "the split arm's partial planes are f32"));
    }
    let mut extra = vec![split.part_out.arg_mut(), split.part_lse.arg_mut()];
    if let Some(lse) = lse {
        extra.push(lse_plane(OP, lse, shape)?);
    }
    let fire = decode_entry(
        OP,
        tactic,
        if lse.is_some() { "split_lse" } else { "split" },
        head_dim,
    )?;
    let mut args = args(OP, q, pool, tables, shape, sm_scale, o, &extra)?;
    args.extend([
        stated(OP, slices)?.arg(),
        stated(OP, shape.rows)?.arg(),
        stated(OP, tactic.split_min_keys.max(1))?.arg(),
    ]);
    ctx.fire(fire.groups([shape.q_heads, shape.rows, slices]), &args)?;

    let mut fold = vec![split.part_out.arg(), split.part_lse.arg(), o.arg_mut()];
    if let Some(lse) = lse {
        fold.push(lse.arg_mut());
    }
    fold.extend([
        stated(OP, slices)?.arg(),
        stated(OP, shape.rows)?.arg(),
        stated(OP, shape.q_heads)?.arg(),
        stated(OP, head_dim)?.arg(),
    ]);
    let entry = if lse.is_some() {
        "attn_sdpa_fold_lse_bf16"
    } else {
        "attn_sdpa_fold_bf16"
    };
    ctx.fire(
        Fire::at(SDPA, entry).apply(Grid::of([o.width / 2, shape.rows, 1], [FOLD_GROUP, 1, 1])),
        &fold,
    )
}

/// Each query row attends the key blocks `selection` names (block ids
/// `ratio` keys wide, `selection.width` a row, a negative id or one past the
/// row's whole blocks skipped) and then its open block, causal.
pub fn selected(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    q: Tensor,
    tables: &Tables,
    pool: &KvPool,
    head_dim: u32,
    sm_scale: f32,
    ratio: u32,
    selection: Tensor,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.selected";
    let shape = shape(OP, q, pool, None, head_dim)?;
    if selection.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the selection is {:?}, and the reader walks i32 block ids",
                selection.dtype
            ),
        ));
    }
    let ratio = nonzero(OP, "the block width this reader expands", ratio)?;
    let top_k = nonzero(OP, "the selection budget", selection.width)?;
    if selection.rows < shape.rows {
        return Err(refuse(
            OP,
            format!(
                "the selection seats {} rows and this reader launches {}",
                selection.rows, shape.rows
            ),
        ));
    }
    let fire = decode_entry(OP, tactic, "selected", head_dim)?;
    let mut args = args(OP, q, pool, tables, &shape, sm_scale, o, &[selection.arg()])?;
    args.extend([stated(OP, top_k)?.arg(), stated(OP, ratio)?.arg()]);
    ctx.fire(fire.groups([shape.q_heads, shape.rows, 1]), &args)
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
    let extra = match lse {
        Some(lse) => vec![lse_plane(OP, lse, &shape)?],
        None => Vec::new(),
    };
    let mut args = args(OP, q, pool, tables, &shape, sm_scale, o, &extra)?;
    args.push(stated(OP, shape.rows)?.arg());
    ctx.fire(
        Fire::at(SDPA, entry)
            .groups([shape.q_heads, shape.rows.div_ceil(tile), 1])
            .group([32, tile, 1]),
        &args,
    )
}
