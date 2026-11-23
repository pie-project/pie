//! Paged attention over a KV cache (`attn.decode`, `attn.prefill`, their
//! LSE forms, and the selected forms) from `kernels/attn/sdpa.slang`.
//! Decode walks a row's keys with the workgroup's subgroups on the subgroup
//! tier or 32-lane strips on the baseline, a long row split into key ranges
//! folded by their LSEs where the device would otherwise idle; prefill tiles
//! query rows of one head and walks their keys in step, a 16-lane row
//! folding by shuffles on the subgroup tier.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{
    Arg, ArgValue, Ctx, Fire, Grid, Target, Tier, instantiated, nonzero, refuse, stated,
};
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
    /// The device's tier. At `Tier::Subgroup` decode reduces with subgroup
    /// operations on SPIR-V (WGSL's decode is the baseline's), and the
    /// 16-lane prefill tile folds its rows by shuffles (subgroups of at
    /// least 16 lanes); the 32-lane tiles run at the baseline.
    pub tier: Tier,
    /// The device's subgroups run narrower than 32 lanes.
    pub narrow: bool,
    /// Invocations a decode workgroup walks with: 256, or 128.
    pub decode_threads: u32,
    /// Query rows a prefill workgroup holds: 32, 16 or 8 at 32 lanes a row.
    pub tile_rows: u32,
    /// Lanes a prefill row walks with: 32, or 16 (16 rows a workgroup,
    /// whatever `tile_rows` says).
    pub tile_lanes: u32,
    /// Decode workgroups the device wants in flight: a decode with fewer
    /// (heads x rows) splits its rows' keys to reach them.
    pub split_groups: u32,
    /// The fewest keys a split walks.
    pub split_min_keys: u32,
    /// The rows a prefill dispatch holds, times the keys each may attend; a
    /// fire past it takes more dispatches, each costing a few ms. 0 holds a
    /// fire's rows in one dispatch.
    pub prefill_work: u64,
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
    /// The fire's mask plane, a byte a column, whole: each masked row's
    /// columns lie where its entry says, so no row of a window cuts it.
    pub mask: Tensor,
    /// Two u32 a row: the key its columns start at (its lane's committed
    /// boundary; [`MASK_NONE`], past every key, where the row states no
    /// mask), then the byte its columns start at in the plane. Every key
    /// below the boundary is read under the model's own bounds and no mask
    /// byte; a row stating none reads no plane. The masks are data: a fire
    /// of masked and unmasked rows runs the kernels a fire of either runs.
    pub mask_words: Tensor,
    /// The row of the tables, the queries and the output the window's first
    /// row is: they are bound whole, the same views whatever window a fire
    /// cuts, and the kernels add it to each row they index.
    pub row0: u32,
    /// The window's rows, which the kernels launch over.
    pub rows: u32,
}

/// A row's boundary stating no mask: past every key (`sdpa.slang`'s
/// `MASK_NONE`).
pub const MASK_NONE: u32 = u32::MAX >> 1;

impl Tables {
    /// `attn.plan_decode` and `attn.plan_prefill`: the fire's tables, refused
    /// where a kernel would read them as something else.
    pub fn plan(op: &'static str, tables: Tables) -> Result<Tables, Error> {
        let stated = [
            ("position table", tables.positions, Dtype::I32),
            ("owning-request table", tables.request_of_token, Dtype::I32),
            ("mask planes", tables.mask, Dtype::U8),
            ("mask words", tables.mask_words, Dtype::U32),
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
        if tables.mask_words.width != 2 {
            return Err(refuse(
                op,
                format!(
                    "the fire's mask words are {} a row, and a row's entry is two: its \
                     boundary and its offset",
                    tables.mask_words.width
                ),
            ));
        }
        let end = u64::from(tables.row0) + u64::from(tables.rows);
        if end > u64::from(tables.positions.rows) || end > u64::from(tables.mask_words.rows) {
            return Err(refuse(
                op,
                format!(
                    "the window's rows {}..{end} pass the fire tables' {} positions and {}                      mask entries",
                    tables.row0, tables.positions.rows, tables.mask_words.rows
                ),
            ));
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
    row0: u32,
    window: i32,
}

fn shape(
    op: &'static str,
    q: Tensor,
    tables: &Tables,
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
    let rows = nonzero(op, "rows", tables.rows)?;
    if u64::from(tables.row0) + u64::from(rows) > u64::from(q.rows) {
        return Err(refuse(
            op,
            format!(
                "the window's rows {}..{} pass the {} query rows bound",
                tables.row0,
                u64::from(tables.row0) + u64::from(rows),
                q.rows
            ),
        ));
    }
    Ok(Shape {
        q_heads,
        kv_heads,
        rows,
        row0: tables.row0,
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
    if pool.page_stride[0] != pool.page_stride[1] {
        return Err(refuse(
            op,
            format!(
                "the pool's key and value pages are {} and {} words apart, where one page \
                 row addresses both",
                pool.page_stride[0], pool.page_stride[1]
            ),
        ));
    }
    if u64::from(shape.row0) + u64::from(shape.rows) > u64::from(o.rows) {
        return Err(refuse(
            op,
            format!(
                "the window's rows {}..{} pass the {} output rows bound",
                shape.row0,
                u64::from(shape.row0) + u64::from(shape.rows),
                o.rows
            ),
        ));
    }
    let mut args = vec![
        q.arg(),
        o.arg_mut(),
        tables.positions.arg(),
        tables.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        tables.mask.arg(),
        tables.mask_words.arg(),
    ];
    args.extend_from_slice(extra);
    if let Some(ring) = &pool.ring {
        args.extend([ring.lanes.arg(), ring.slots.arg()]);
    }
    args.extend(pool.pool.args(false));
    args.extend([
        pool.pool.words.arg(),
        pool.origin[0].arg(),
        pool.origin[1].arg(),
        pool.page_stride[0].arg(),
        stated(op, shape.q_heads / shape.kv_heads)?.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        shape.window.arg(),
        stated(op, shape.q_heads)?.arg(),
        shape.row0.arg(),
    ]);
    Ok(args)
}

/// The ring's scalars, after every other a form reads (`Params`' last).
fn ring_params(op: &'static str, pool: &KvPool) -> Result<Vec<ArgValue>, Error> {
    let Some(ring) = &pool.ring else {
        return Ok(Vec::new());
    };
    if ring.lanes.dtype != Dtype::I32 || ring.slots.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            "a ring's lane and record tables are i32".to_string(),
        ));
    }
    if ring.pieces == 0 || ring.piece >= ring.pieces || ring.values_piece >= ring.pieces {
        return Err(refuse(
            op,
            format!(
                "the ring lies in piece {} of a record of {} pieces",
                ring.piece, ring.pieces
            ),
        ));
    }
    Ok(vec![
        nonzero(op, "the ring's rows", ring.rows)?.arg(),
        ring.piece.arg(),
        ring.values_piece.arg(),
        ring.pieces.arg(),
        ring.keys.arg(),
        ring.values.arg(),
        ring.slot_stride.arg(),
    ])
}

/// `_ring` where the pool's windowed rows lie in a ring.
fn ring_kind(pool: &KvPool) -> &'static str {
    if pool.ring.is_some() { "_ring" } else { "" }
}

fn lse_plane(op: &'static str, lse: Tensor, shape: &Shape) -> Result<ArgValue, Error> {
    let held = u64::from(lse.rows) * u64::from(lse.width);
    let rows = u64::from(shape.row0) + u64::from(shape.rows);
    if lse.dtype != Dtype::F32 || held < rows * u64::from(shape.q_heads) {
        return Err(refuse(
            op,
            format!(
                "the log-sum-exp plane is {} x {} {:?}, and it holds an f32 a head a row ({} x {})",
                lse.rows, lse.width, lse.dtype, rows, shape.q_heads
            ),
        ));
    }
    Ok(lse.arg_mut())
}

/// The kind a decode's walk of a row's keys is instantiated under: whole or
/// split into ranges, landing the row's LSE or not.
fn walk_kind(split: bool, lse: bool) -> &'static str {
    match (split, lse) {
        (false, false) => "decode",
        (false, true) => "decode_lse",
        (true, false) => "split",
        (true, true) => "split_lse",
    }
}

/// The kernels a decode of `head_dim`-wide heads fires on `target` under
/// `tactic` (`lse` where it lands its LSE): the walk of a row's keys whole,
/// then the walk that splits them. A load holds them to its device before
/// any fire names one.
pub fn decode_kernels(
    target: Target,
    tactic: Tactic,
    head_dim: u32,
    lse: bool,
    ring: bool,
) -> Result<[Fire; 2], Error> {
    let ring = if ring { "_ring" } else { "" };
    Ok([
        decode_entry(
            target,
            "attn.decode",
            tactic,
            &format!("{}{ring}", walk_kind(false, lse)),
            head_dim,
        )?,
        decode_entry(
            target,
            "attn.decode_split",
            tactic,
            &format!("{}{ring}", walk_kind(true, lse)),
            head_dim,
        )?,
    ])
}

/// A decode workgroup's entry: `kind`, then its thread and lane variants.
fn decode_entry(
    target: Target,
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
    let subgroup = tactic.tier == Tier::Subgroup && target == Target::Spirv;
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
    let shape = shape(OP, q, tables, pool, window, head_dim)?;
    if let Some(names) = ctx.naming() {
        // The rows and the keys a fire holds say whether it splits: where
        // the planes are there to split into, either walk, then the fold.
        let [whole, ranges] = decode_kernels(
            ctx.target(),
            tactic,
            head_dim,
            lse.is_some(),
            pool.ring.is_some(),
        )?;
        if split.is_none() {
            return names.either(&[whole]);
        }
        names.either(&[whole, ranges])?;
        return names.either(&[fold_kernel(lse.is_some())]);
    }
    if let Some(planes) = split {
        let ceiling = split_ceiling(&planes, &shape, head_dim);
        // The splits are the shape's alone, never the keys the fire holds:
        // each row's walker splits its own keys (no split under the fewest,
        // and one split lands the row whole for the fold to pass), so a
        // fire's dispatches are the same as its keys grow.
        let slices = splits_for(&tactic, &shape, ceiling);
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
        ctx.target(),
        OP,
        tactic,
        &format!("{}{}", walk_kind(false, lse.is_some()), ring_kind(pool)),
        head_dim,
    )?;
    let mut args = args(OP, q, pool, tables, &shape, sm_scale, o, &extra)?;
    args.extend(ring_params(OP, pool)?);
    ctx.fire(fire.groups([shape.q_heads, shape.rows, 1]), &args)
}

/// The splits a decode of this shape takes: enough (heads x rows x splits)
/// workgroups to reach the device's want, and no more than the partial
/// planes hold. Its rows' keys play no part: each row's walker takes as many
/// of these as its own keys fill (`split_min_keys` each), the rest landing
/// nothing.
fn splits_for(tactic: &Tactic, shape: &Shape, ceiling: u32) -> u32 {
    let groups = shape.q_heads.saturating_mul(shape.rows).max(1);
    if ceiling <= 1 || groups >= tactic.split_groups {
        return 1;
    }
    tactic
        .split_groups
        .div_ceil(groups)
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
        ctx.target(),
        OP,
        tactic,
        &format!("{}{}", walk_kind(true, lse.is_some()), ring_kind(pool)),
        head_dim,
    )?;
    let mut args = args(OP, q, pool, tables, shape, sm_scale, o, &extra)?;
    args.extend([
        stated(OP, slices)?.arg(),
        stated(OP, shape.rows)?.arg(),
        stated(OP, tactic.split_min_keys.max(1))?.arg(),
    ]);
    args.extend(ring_params(OP, pool)?);
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
        shape.row0.arg(),
    ]);
    ctx.fire(
        fold_kernel(lse.is_some())
            .apply(Grid::of([o.width / 2, shape.rows, 1], [FOLD_GROUP, 1, 1])),
        &fold,
    )
}

/// The kernel that weighs a split decode's partials into its rows, landing
/// their LSE where `lse`.
fn fold_kernel(lse: bool) -> Fire {
    Fire::at(
        SDPA,
        if lse {
            "attn_sdpa_fold_lse_bf16"
        } else {
            "attn_sdpa_fold_bf16"
        },
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
    if pool.ring.is_some() {
        return Err(refuse(
            OP,
            "a selection reads keys by score, and a ring is a window's".to_string(),
        ));
    }
    let shape = shape(OP, q, tables, pool, None, head_dim)?;
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
    if u64::from(selection.rows) < u64::from(shape.row0) + u64::from(shape.rows) {
        return Err(refuse(
            OP,
            format!(
                "the selection seats {} rows and this reader reads rows {}..{}",
                selection.rows,
                shape.row0,
                u64::from(shape.row0) + u64::from(shape.rows)
            ),
        ));
    }
    let fire = decode_entry(ctx.target(), OP, tactic, "selected", head_dim)?;
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
    let shape = shape(OP, q, tables, pool, window, head_dim)?;
    let (lanes, tile) = match (tactic.tile_lanes, tactic.tile_rows) {
        (..=16, _) => (16, 16),
        (_, 32..) => (32, 32),
        (_, 16..=31) => (32, 16),
        _ => (32, 8),
    };
    // A pool of one segment (every page in it) runs the `_seg1` form, which
    // reads it with no copy of the loads a segment; a commit of another
    // segment forgets every recording, so a fire's form follows its pool.
    // SPIR-V only: WGSL drops the unread segment bindings, and its chain an
    // element reads the one segment as fast.
    let one_segment = ctx.target() == Target::Spirv
        && pool.pool.bufs[1..].iter().all(|&buf| buf == crate::ABSENT);
    let name = format!(
        "attn_sdpa_{}{}{}{}{}_bf16_d_{head_dim}",
        if lse.is_some() { "tiled_lse" } else { "tiled" },
        ring_kind(pool),
        if lanes == 32 { "" } else { "_l16" },
        if tile == 32 {
            String::new()
        } else {
            format!("_r{tile}")
        },
        if one_segment { "_seg1" } else { "" },
    );
    let entry = instantiated(OP, name)?;
    let tier = if lanes == 16 && tactic.tier == Tier::Subgroup {
        Tier::Subgroup
    } else {
        Tier::Baseline
    };
    let extra = match lse {
        Some(lse) => vec![lse_plane(OP, lse, &shape)?],
        None => Vec::new(),
    };
    let mut args = args(OP, q, pool, tables, &shape, sm_scale, o, &extra)?;
    args.push(stated(OP, shape.rows)?.arg());
    let ring = ring_params(OP, pool)?;
    let keys = u64::from(pool.page_indices.rows) * pool.page_size.max(1) as u64;
    let keys = window
        .map_or(keys, |window| keys.min(u64::from(window)))
        .max(1);
    // A naming sink is told the longest way: at the most rows and keys a
    // fire of the load may bring.
    let (rows, keys) = match ctx.naming() {
        Some(names) => (names.most_rows().max(shape.rows), Keys::Most),
        None => (shape.rows, Keys::Held(keys)),
    };
    let tiles = rows.div_ceil(tile);
    let step = prefill_step(tactic.prefill_work, rows, tile, keys);
    for first in (0..tiles).step_by(step as usize) {
        let mut args = args.clone();
        args.push(stated(OP, first)?.arg());
        args.extend_from_slice(&ring);
        ctx.fire(
            Fire::at(SDPA, entry)
                .tier(tier)
                .groups([shape.q_heads, step.min(tiles - first), 1])
                .group([lanes, tile, 1]),
            &args,
        )?;
    }
    Ok(())
}

/// The keys a prefill's rows attend: what the fire's tables hold, or the
/// most a fire may bring, which a naming sink is told.
#[derive(Clone, Copy, Debug)]
enum Keys {
    Held(u64),
    Most,
}

/// Tiles of `tile` rows one prefill dispatch holds, of a fire's `rows`:
/// every one where `work` (rows times keys a dispatch may take; 0 for no
/// bound) holds the fire's rows at `keys`, else as many whole tiles as it
/// holds rows, one at least. At the most keys a fire may bring a budget past
/// 0 holds one tile a dispatch: the longest way its rows can take.
fn prefill_step(work: u64, rows: u32, tile: u32, keys: Keys) -> u32 {
    let tiles = rows.div_ceil(tile).max(1);
    let held = match (work, keys) {
        (0, _) => return tiles,
        (_, Keys::Most) => return 1,
        (work, Keys::Held(keys)) => work / keys.max(1),
    };
    if held >= u64::from(rows) {
        tiles
    } else {
        (held as u32 / tile).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dispatches a prefill of `rows` makes at `step` tiles each.
    fn dispatches(rows: u32, tile: u32, step: u32) -> u32 {
        rows.div_ceil(tile).max(1).div_ceil(step)
    }

    /// With no work budget a prefill holds its rows in one dispatch, at
    /// every row count and keys, a tile filled or not; with one, the
    /// dispatches grow with the rows and never past what a naming sink is
    /// told, the longest way the rows can take.
    #[test]
    fn a_prefill_dispatches_as_its_budget_says_and_names_its_longest_way() {
        for tile in [8, 16, 32] {
            for rows in 1..=4096 {
                for keys in [1, 512, 40_960] {
                    let step = prefill_step(0, rows, tile, Keys::Held(keys));
                    assert_eq!(dispatches(rows, tile, step), 1, "{rows} rows of {tile}");
                }
                assert_eq!(
                    dispatches(rows, tile, prefill_step(0, rows, tile, Keys::Most)),
                    1
                );
            }
        }
        let work = 1 << 20;
        for tile in [8, 16, 32] {
            for keys in [1u64, 64, 512, 4096, 40_960, 1 << 21] {
                let mut before = 0;
                for rows in 1..=4096 {
                    let made =
                        dispatches(rows, tile, prefill_step(work, rows, tile, Keys::Held(keys)));
                    let named = dispatches(rows, tile, prefill_step(work, rows, tile, Keys::Most));
                    assert!(
                        made >= before,
                        "{rows} rows at {keys} keys: {made} after {before}"
                    );
                    assert!(
                        made <= named,
                        "{rows} rows at {keys} keys: {made} past {named}"
                    );
                    before = made;
                }
            }
        }
    }

    /// A naming sink of the most rows a fire may bring, counting what is
    /// fired into it.
    struct Counted {
        most: u32,
        fired: std::cell::Cell<u32>,
    }

    impl crate::Encode for Counted {
        fn fire(&self, _fire: Fire, _args: &[ArgValue]) -> Result<(), Error> {
            self.fired.set(self.fired.get() + 1);
            Ok(())
        }

        fn absent(&self) -> Result<ArgValue, Error> {
            Ok(ArgValue::Buffer(crate::ABSENT))
        }

        fn target(&self) -> Target {
            Target::Wgsl
        }

        fn naming(&self) -> Option<&dyn crate::Naming> {
            Some(self)
        }
    }

    impl crate::Naming for Counted {
        fn either(&self, _alternatives: &[Fire]) -> Result<(), Error> {
            self.fired.set(self.fired.get() + 1);
            Ok(())
        }

        fn fork(&self) {}

        fn or(&self) {}

        fn join(&self) {}

        fn most_rows(&self) -> u32 {
            self.most
        }
    }

    /// A budgeted prefill a naming sink walks counts at the most rows the
    /// sink states, a tile a dispatch, whatever rows the naming fire holds;
    /// unbudgeted, one dispatch. The negative control: a prefill of the
    /// naming fire's own rows makes fewer.
    #[test]
    fn a_budgeted_prefill_counts_at_the_most_rows_a_naming_sink_states() {
        let tactic = |prefill_work| Tactic {
            tier: Tier::Baseline,
            narrow: false,
            decode_threads: 256,
            tile_rows: 16,
            tile_lanes: 16,
            split_groups: 1,
            split_min_keys: 1,
            prefill_work,
        };
        let tensor = |buf, rows, width, dtype| Tensor::new(buf, rows, width, dtype);
        let (heads, head_dim, staged, most) = (2u32, 64u32, 3u32, 4096u32);
        let q = tensor(1, staged, heads * head_dim, Dtype::Bf16);
        let o = tensor(2, staged, heads * head_dim, Dtype::Bf16);
        let tables = Tables {
            positions: tensor(3, staged, 1, Dtype::I32),
            request_of_token: tensor(4, staged, 1, Dtype::I32),
            mask: tensor(5, 1, 1, Dtype::U8),
            mask_words: tensor(6, staged, 2, Dtype::U32),
            row0: 0,
            rows: staged,
        };
        let plane = tensor(7, 64, heads * head_dim, Dtype::Bf16);
        let pool = KvPool::planes(
            plane,
            plane,
            tensor(8, 4, 1, Dtype::I32),
            tensor(9, 2, 1, Dtype::I32),
            16,
            u64::from(heads * head_dim),
            u64::from(head_dim),
        );
        let fired = |prefill_work, most| {
            let sink = Counted {
                most,
                fired: std::cell::Cell::new(0),
            };
            prefill(
                &sink,
                tactic(prefill_work),
                q,
                &tables,
                &pool,
                None,
                head_dim,
                1.0,
                o,
                None,
            )
            .unwrap_or_else(|why| panic!("the prefill names its fires: {why}"));
            sink.fired.get()
        };
        assert_eq!(fired(1 << 20, most), most.div_ceil(16));
        assert_eq!(fired(0, most), 1);
        assert_eq!(fired(1 << 20, 0), staged.div_ceil(16));
        assert!(fired(1 << 20, 0) < fired(1 << 20, most));
    }

    /// A decode's kernels are named before any fire, as the fire names
    /// them: the whole walk and the split one, their LSE forms, at the
    /// tactic's threads; a head width no kernel is built at names none.
    #[test]
    fn a_decodes_kernels_are_named_without_a_fire() {
        let tactic = Tactic {
            tier: Tier::Baseline,
            narrow: false,
            decode_threads: 256,
            tile_rows: 16,
            tile_lanes: 16,
            split_groups: 1,
            split_min_keys: 1,
            prefill_work: 0,
        };
        let names = |tactic, head_dim, lse| {
            decode_kernels(Target::Wgsl, tactic, head_dim, lse, false)
                .map(|fires| fires.map(|fire| (fire.file, fire.entrypoint, fire.tier)))
        };
        assert_eq!(
            names(tactic, 64, false).unwrap(),
            [
                (SDPA, "attn_sdpa_decode_bf16_d_64", Tier::Baseline),
                (SDPA, "attn_sdpa_split_bf16_d_64", Tier::Baseline),
            ]
        );
        let narrow = Tactic {
            decode_threads: 128,
            ..tactic
        };
        assert_eq!(
            names(narrow, 128, true).unwrap(),
            [
                (SDPA, "attn_sdpa_decode_lse_t128_bf16_d_128", Tier::Baseline),
                (SDPA, "attn_sdpa_split_lse_t128_bf16_d_128", Tier::Baseline),
            ]
        );
        assert!(names(tactic, 96, false).is_err());
    }
}
