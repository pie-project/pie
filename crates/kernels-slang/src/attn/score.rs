//! Score capture (`attn.score_capture`, `attn.score_pages`) from
//! `kernels/attn/score.slang`: each capturing lane's one f32 row, a mass a
//! key of the load's context, its causal softmax over its paged keys summed
//! over this layer's query heads and averaged over its last `observe` query
//! rows and the model's query heads, added into the row or into zero; and
//! once, the final row summed a kv page.
//!
//! A fire's capturing lanes are compacted ([`TABLE_INTS`] a lane, in lane
//! order), each lane's rows addressed through the bases its entry states.
//! Every head and observed row is summed inside the dispatch, so no
//! per-head row is laid, and in one order a key, so a run reproduces. Two
//! dispatches a layer, wide over the keys: the normalisers of each (lane,
//! head, observed row) into a scratch of [`stats_floats`], then the mass, a
//! thread a key.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, Tensor};

const SCORE: &str = "attn/score.slang";

const THREADS: u32 = 256;

const STAMPS: [u32; 3] = [64, 128, 256];

const NORM: [&str; 3] = [
    "attn_score_norm_bf16_d_64",
    "attn_score_norm_bf16_d_128",
    "attn_score_norm_bf16_d_256",
];

fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

/// The mass kernel's forms by the workgroup memory they take, each its
/// tile's key rows' and query rows' words (`PIE_MASS_ROWS`, `PIE_MASS_Q`):
/// a device states which it runs ([`mass_form`]).
const MASS: [MassForm; 2] = [
    MassForm {
        entry: "attn_score_mass_s32",
        rows: 5120,
        q: 2112,
        bytes: 32 << 10,
    },
    MassForm {
        entry: "attn_score_mass_s16",
        rows: 2080,
        q: 1088,
        bytes: 16 << 10,
    },
];

/// The query heads the mass kernel stages a normaliser of (`PIE_MASS_HEADS`).
const MASS_HEADS: u32 = 64;

/// The workgroups a capturing lane's mass runs over (`PIE_MASS_GRID`): the
/// tiles past them are walked by the same.
const MASS_GRID: u32 = 1024;

#[derive(Clone, Copy, Debug)]
struct MassForm {
    entry: &'static str,
    rows: u32,
    q: u32,
    bytes: u32,
}

/// The mass kernel's form on a device of `workgroup_bytes` of workgroup
/// memory, and its keys a tile: the widest that fits the device and
/// `heads` query heads over `kv_heads` of `head_dim`, every kv head's row of
/// a tile's keys and every query head's row staged at once.
fn mass_form(workgroup_bytes: u32, heads: u32, kv_heads: u32, head_dim: u32) -> Option<(MassForm, u32)> {
    let run = head_dim / 2 + 1;
    MASS.into_iter().find_map(|form| {
        let fits = workgroup_bytes >= form.bytes
            && heads <= MASS_HEADS
            && heads.saturating_mul(run) <= form.q;
        let tile = (form.rows / kv_heads.saturating_mul(run).max(1)).min(THREADS / heads.max(1));
        (fits && tile > 0).then_some((form, tile))
    })
}

const PAGES: &str = "attn_score_pages";

/// The ints a capturing lane's entry of the capture table holds: the fire
/// lane, its first and end query row in the fire, and the f32 its key row
/// starts at in the key rows and its page row in the page rows (-1 where
/// its program reads no pages). A lane's rows are addressed through those
/// two bases alone.
pub const TABLE_INTS: u32 = 5;

/// The f32s the normalisers of `capturing` lanes take, `heads` query heads
/// of this rank each and `observe` rows a head: a maximum and an inverse
/// sum each.
#[must_use]
pub fn stats_floats(capturing: u32, heads: u32, observe: u32) -> u64 {
    u64::from(capturing) * u64::from(heads) * u64::from(observe) * 2
}

/// One layer's capture over a fire's capturing lanes.
#[derive(Clone, Copy, Debug)]
pub struct Capture {
    /// The fire's query rows, every lane's (bf16, this rank's heads).
    pub q: Tensor,
    /// The fire's absolute position a query row (i32).
    pub positions: Tensor,
    /// The capture table: [`TABLE_INTS`] i32 a capturing lane.
    pub table: Tensor,
    /// The capturing lanes: the rows of `scores`.
    pub capturing: u32,
    /// The normalisers' scratch, at least [`stats_floats`] f32.
    pub stats: Tensor,
    /// The key rows, `[capturing, context]` f32: added into, or written.
    pub scores: Tensor,
}

/// The kv heads the pool row's strides spell, which must be the stated ones.
fn kv_heads_agree(
    op: &'static str,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
) -> Result<(), Error> {
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
    let spelled = pool.seq_stride / pool.head_stride;
    if u64::from(kv_heads) != spelled {
        return Err(refuse(
            op,
            format!(
                "the stated kv head count {kv_heads} is not the {spelled} the pool row's strides spell"
            ),
        ));
    }
    Ok(())
}

pub fn capture(
    ctx: &Ctx<'_>,
    at: &Capture,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    q_heads: u32,
    sm_scale: f32,
    observe: u32,
    into: bool,
    workgroup_bytes: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.score_capture";
    let q = at.q;
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
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
    for (what, tensor) in [("the score rows", at.scores), ("the normalisers' scratch", at.stats)] {
        if tensor.dtype != Dtype::F32 {
            return Err(refuse(
                OP,
                format!("{what} are {:?}, and a mass is an f32", tensor.dtype),
            ));
        }
    }
    for (what, tensor) in [("position table", at.positions), ("capture table", at.table)] {
        if tensor.dtype != Dtype::I32 {
            return Err(refuse(
                OP,
                format!("the fire's {what} is {:?}, and this capture reads i32", tensor.dtype),
            ));
        }
    }
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let Some(stamp) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide dot the normaliser is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    if !head_dim.is_multiple_of(16) {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is read eight bf16 pairs at a time, and is no multiple of 16"
            ),
        ));
    }
    let heads = super::row_heads(OP, q.width, head_dim)?;
    if !heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            OP,
            format!("{heads} query heads do not group over {kv_heads} kv heads"),
        ));
    }
    let Some((mass, tile)) = mass_form(workgroup_bytes, heads, kv_heads, head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "{heads} query heads over {kv_heads} kv heads of {head_dim}: no form of the mass \
                 kernel stages a key's rows and the heads' query rows in the {workgroup_bytes} \
                 bytes of workgroup memory this device states"
            ),
        ));
    };
    if q_heads < heads {
        return Err(refuse(
            OP,
            format!(
                "the model states {q_heads} query heads to average over, fewer than the \
                 {heads} this rank's row holds"
            ),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    nonzero(OP, "the capturing lanes this capture answers", at.capturing)?;
    let page_size = u32::try_from(pool.page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| refuse(OP, "the kv page size is zero"))?;
    let context = at.scores.width;
    nonzero(OP, "the keys of the load's context", context)?;
    if at.scores.rows < at.capturing {
        return Err(refuse(
            OP,
            format!(
                "the score rows hold {} capturing lanes, and {} capture",
                at.scores.rows, at.capturing
            ),
        ));
    }
    if at.table.rows.saturating_mul(at.table.width) < at.capturing.saturating_mul(TABLE_INTS) {
        return Err(refuse(
            OP,
            format!(
                "the capture table holds {} ints, and {} capturing lanes take {TABLE_INTS} each",
                at.table.rows.saturating_mul(at.table.width),
                at.capturing
            ),
        ));
    }
    let tiles = context.div_ceil(tile);
    let launches = |what: &str, groups: u64| {
        u32::try_from(groups).map_err(|_| {
            refuse(OP, format!("the {what} grid will not launch: {groups} workgroups"))
        })
    };
    let norm_x = launches("normaliser", u64::from(at.capturing) * u64::from(THREADS))?;
    let mass_x = launches("mass", u64::from(tiles.min(MASS_GRID)) * u64::from(THREADS))?;

    let scalars = |into: bool, tile: u32| -> Result<Vec<ArgValue>, Error> {
        Ok(vec![
            pool.pool.words.arg(),
            pool.origin[0].arg(),
            pool.page_stride[0].arg(),
            stated(OP, page_size)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, context)?.arg(),
            i32::from(into).arg(),
            stated(OP, q_heads)?.arg(),
            stated(OP, tile)?.arg(),
        ])
    };
    let reads = [
        q.arg(),
        at.table.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        at.positions.arg(),
    ];
    let need = stats_floats(at.capturing, heads, observe);
    let have = u64::from(at.stats.rows) * u64::from(at.stats.width);
    if have < need {
        return Err(refuse(
            OP,
            format!("the normalisers take {need} f32, and the scratch holds {have}"),
        ));
    }
    ctx.fire(
        Fire::at(SCORE, NORM[stamp]).apply(Grid::of([norm_x, heads, 1], [THREADS, 1, 1])),
        &[
            &reads[..],
            &[at.stats.arg_mut()],
            &pool.pool.args(false),
            &scalars(into, tile)?,
        ]
        .concat(),
    )?;
    ctx.fire(
        Fire::at(SCORE, mass.entry).apply(Grid::of([mass_x, at.capturing, 1], [THREADS, 1, 1])),
        &[
            &reads[..],
            &[at.stats.arg(), at.scores.arg_mut()],
            &pool.pool.args(false),
            &scalars(into, tile)?,
        ]
        .concat(),
    )
}

/// The final key rows `keys` (`[capturing, context]`) summed a kv page of
/// `page_size` keys into `pages` (`[page rows, context pages]`): each
/// capturing lane whose table entry names a page row lands it there.
pub fn pages(
    ctx: &Ctx<'_>,
    table: Tensor,
    capturing: u32,
    keys: Tensor,
    pages: Tensor,
    page_size: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.score_pages";
    for (what, tensor) in [("key rows", keys), ("page rows", pages)] {
        if tensor.dtype != Dtype::F32 {
            return Err(refuse(
                OP,
                format!("the {what} are {:?}, and a mass is an f32", tensor.dtype),
            ));
        }
    }
    if table.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the capture table is {:?}, and this sum reads i32", table.dtype),
        ));
    }
    nonzero(OP, "the capturing lanes this sum answers", capturing)?;
    nonzero(OP, "the kv page size", page_size)?;
    let context = keys.width;
    if pages.width != context.div_ceil(page_size) {
        return Err(refuse(
            OP,
            format!(
                "the page rows are {} wide, and {context} keys in pages of {page_size} are {}",
                pages.width,
                context.div_ceil(page_size)
            ),
        ));
    }
    if keys.rows < capturing {
        return Err(refuse(
            OP,
            format!("the key rows hold {} lanes, and {capturing} capture", keys.rows),
        ));
    }
    let x = u32::try_from(u64::from(pages.width.div_ceil(THREADS)) * u64::from(THREADS))
        .map_err(|_| refuse(OP, "the page grid will not launch"))?;
    ctx.fire(
        Fire::at(SCORE, PAGES).apply(Grid::of([x, capturing, 1], [THREADS, 1, 1])),
        &[
            table.arg(),
            keys.arg(),
            pages.arg_mut(),
            stated(OP, context)?.arg(),
            stated(OP, pages.width)?.arg(),
            stated(OP, page_size)?.arg(),
        ],
    )
}
