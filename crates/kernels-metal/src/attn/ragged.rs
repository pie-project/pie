use dtype::Dtype;

use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "attn/ragged.metal";

const OP: &str = "attention.ragged";

const MMA_THREADS: u32 = 128;

const MMA_TILE: u32 = 32;

const SCALAR_THREADS: u32 = 128;

#[derive(Clone, Copy, Debug)]
pub enum RaggedMask {
    Segments,
    ReferenceTags { q_tags: Tensor, kv_tags: Tensor },
    RelativeBias { table: Tensor, max_len: u32 },
}

impl RaggedMask {
    const fn suffix(&self) -> &'static str {
        match self {
            Self::Segments => "none",
            Self::ReferenceTags { .. } => "tags",
            Self::RelativeBias { .. } => "bias",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Tiled,
    Scalar,
}

fn stamp(arm: Arm, head_dim: u32) -> Result<&'static str, Error> {
    match arm {
        Arm::Tiled => match head_dim {
            64 => Ok("64"),
            128 => Ok("128"),
            other => Err(refuse(
                OP,
                format!(
                    "no tiled point is stamped at head width {other}; the tile is stamped at \
                     64 and 128, and a width the matrix unit cannot tile takes the scalar arm"
                ),
            )),
        },
        Arm::Scalar => match head_dim {
            0..=32 => Ok("32"),
            33..=64 => Ok("64"),
            65..=128 => Ok("128"),
            129..=256 => Ok("256"),
            other => Err(refuse(
                OP,
                format!(
                    "no point is stamped at head width {other}; the scalar arm walks any \
                     width up to 256, at the smallest of 32, 64, 128 and 256 that holds it"
                ),
            )),
        },
    }
}

struct Shape {
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    segments: u32,
}

#[allow(clippy::too_many_arguments)]
fn shape(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    o: Tensor,
) -> Result<Shape, Error> {
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    let rows = nonzero(OP, "query rows", q.rows)?;
    let kv_heads = nonzero(OP, "kv heads", kv_heads)?;
    for (what, t) in [("query", q), ("key", k), ("value", v), ("answer", o)] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
        if t.width % head_dim != 0 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} rectangle is {} wide, which is not a whole number of \
                     {head_dim}-wide heads",
                    t.width
                ),
            ));
        }
    }
    if k.width != kv_heads * head_dim || v.width != k.width || k.rows != v.rows {
        return Err(refuse(
            OP,
            format!(
                "the key rectangle is {} x {} and the value {} x {}; both are \
                 {kv_heads} x {head_dim}",
                k.rows, k.width, v.rows, v.width
            ),
        ));
    }
    if o.rows != rows || o.width != q.width {
        return Err(refuse(
            OP,
            format!(
                "the answer is {} x {} and the query {rows} x {}",
                o.rows, o.width, q.width
            ),
        ));
    }
    let q_heads = q.width / head_dim;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            OP,
            format!("{q_heads} query heads do not group over {kv_heads} kv heads"),
        ));
    }
    for (what, t) in [("query", q_indptr), ("key", kv_indptr)] {
        if t.dtype != Dtype::I32 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} CSR is {:?}, and a segment table is i32",
                    t.dtype
                ),
            ));
        }
    }
    let entries = (q_indptr.rows * q_indptr.width).min(kv_indptr.rows * kv_indptr.width);
    let segments = entries
        .checked_sub(1)
        .filter(|s| *s > 0)
        .ok_or_else(|| refuse(OP, format!("a CSR of {entries} entries names no segment")))?;
    Ok(Shape {
        rows,
        q_heads,
        kv_heads,
        segments,
    })
}

fn mask_args(ctx: &Ctx<'_>, mask: &RaggedMask, q_heads: u32) -> Result<[ArgValue; 4], Error> {
    match mask {
        RaggedMask::Segments => Ok([ctx.absent()?, ctx.absent()?, ctx.absent()?, 0i32.arg()]),
        RaggedMask::ReferenceTags { q_tags, kv_tags } => {
            for (what, t) in [("query", *q_tags), ("key", *kv_tags)] {
                if t.dtype != Dtype::I32 {
                    return Err(refuse(
                        OP,
                        format!("the {what} tag table is {:?}, and a tag is i32", t.dtype),
                    ));
                }
            }
            Ok([q_tags.arg(), kv_tags.arg(), ctx.absent()?, 0i32.arg()])
        }
        RaggedMask::RelativeBias { table, max_len } => {
            let max_len = nonzero(OP, "the bias table's reach", *max_len)?;
            let span = 2 * max_len - 1;
            if table.dtype != Dtype::F32 {
                return Err(refuse(
                    OP,
                    format!(
                        "the bias table is {:?}, and a logit bias is f32",
                        table.dtype
                    ),
                ));
            }
            if u64::from(table.rows) * u64::from(table.width) < u64::from(q_heads) * u64::from(span)
            {
                return Err(refuse(
                    OP,
                    format!(
                        "the bias table is {} x {}, and this launch reads {span} \
                         (2 x {max_len} - 1) columns for each of {q_heads} query heads",
                        table.rows, table.width
                    ),
                ));
            }
            let stated = i32::try_from(max_len)
                .map_err(|_| refuse(OP, format!("a reach of {max_len} does not fit an int")))?;
            Ok([ctx.absent()?, ctx.absent()?, table.arg(), stated.arg()])
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: &RaggedMask,
    o: Tensor,
) -> Result<(), Error> {
    let scalar = std::env::var("PIE_METAL_RAGGED_ARM")
        .map(|v| v.eq_ignore_ascii_case("scalar"))
        .unwrap_or(false);
    let arm = if !scalar && stamp(Arm::Tiled, head_dim).is_ok() {
        Arm::Tiled
    } else {
        Arm::Scalar
    };
    fire(
        ctx, arm, q, k, v, q_indptr, kv_indptr, head_dim, kv_heads, sm_scale, mask, o,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fire(
    ctx: &Ctx<'_>,
    arm: Arm,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: &RaggedMask,
    o: Tensor,
) -> Result<(), Error> {
    let s = shape(q, k, v, q_indptr, kv_indptr, head_dim, kv_heads, o)?;
    let width = stamp(arm, head_dim)?;
    let m = mask_args(ctx, mask, s.q_heads)?;
    let segments = i32::try_from(s.segments)
        .map_err(|_| refuse(OP, format!("{} segments do not fit an int", s.segments)))?;
    let kv_heads = i32::try_from(s.kv_heads).expect("a checked head count");

    match arm {
        Arm::Tiled => {
            let entry = leaked(format!("ragged_mma_bfloat16_d_{width}_{}", mask.suffix()));
            let lanes = s.q_heads.checked_mul(MMA_THREADS).ok_or_else(|| {
                refuse(
                    OP,
                    format!("the grid will not launch: {} query heads", s.q_heads),
                )
            })?;
            let rows = i32::try_from(s.rows).expect("a checked row count");
            ctx.fire(
                Fire::at(FILE, entry).apply(Grid::of(
                    [lanes, s.rows.div_ceil(MMA_TILE).max(1), 1],
                    [MMA_THREADS, 1, 1],
                )),
                &[
                    q.arg(),
                    k.arg(),
                    v.arg(),
                    o.arg_mut(),
                    q_indptr.arg(),
                    kv_indptr.arg(),
                    segments.arg(),
                    kv_heads.arg(),
                    sm_scale.arg(),
                    m[0],
                    m[1],
                    m[2],
                    m[3],
                    rows.arg(),
                ],
            )
        }
        Arm::Scalar => {
            let entry = leaked(format!(
                "ragged_scalar_bfloat16_d_{width}_{}",
                mask.suffix()
            ));
            let lanes = s.q_heads.checked_mul(SCALAR_THREADS).ok_or_else(|| {
                refuse(
                    OP,
                    format!("the grid will not launch: {} query heads", s.q_heads),
                )
            })?;
            let q_heads = i32::try_from(s.q_heads).expect("a checked head count");
            let head_dim = i32::try_from(head_dim).expect("a checked head width");
            ctx.fire(
                Fire::at(FILE, entry).apply(Grid::of([lanes, s.rows, 1], [SCALAR_THREADS, 1, 1])),
                &[
                    q.arg(),
                    k.arg(),
                    v.arg(),
                    o.arg_mut(),
                    q_indptr.arg(),
                    kv_indptr.arg(),
                    segments.arg(),
                    q_heads.arg(),
                    kv_heads.arg(),
                    head_dim.arg(),
                    sm_scale.arg(),
                    m[0],
                    m[1],
                    m[2],
                    m[3],
                ],
            )
        }
    }
}

fn leaked(name: String) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut set = NAMES
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .expect("the entry-name table");
    if let Some(found) = set.get(name.as_str()) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.into_boxed_str());
    set.insert(leaked);
    leaked
}
