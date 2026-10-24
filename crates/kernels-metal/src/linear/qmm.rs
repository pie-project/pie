//! A projection over a quantized bank (`linear.matmul`, `linear.matmul`):
//! the profile's `kernel.qmm` ladder picks a kernel for the shape, each rung
//! running if it finds its scratch. The matrix kernels are this module's
//! (`mpp`, `staged`, `sg`, `tiled`, one per `linear/qmm_*.metal` unit); the
//! vector ones are [`super::qmv`]'s.

pub mod mpp;
pub mod sg;
pub mod staged;
pub mod tiled;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;

use crate::Tuning;
use crate::encode::{Ctx, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};
use crate::tuning::Qmm;

use super::qmv;

pub(crate) const GROUPS: [i32; 3] = [32, 64, 128];

pub(crate) const WIDTHS: [i32; 3] = [2, 4, 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub entry: &'static str,

    pub stamp: &'static str,
}

pub(crate) fn check(
    op: &'static str,
    points: &[i32],
    v: i32,
    what: &'static str,
) -> Result<(), Error> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is stamped at {what} {v}")))
}

pub(crate) fn symbol(name: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}

/// The register tile's input table for a matmul's activation: given the
/// table's rows, contraction and arrival counters, its room and whether it
/// already stands there (see [`Scratch::table`]).
pub type KeptTable<'a> = &'a dyn Fn(u32, u32, u32) -> Option<(Tensor, bool)>;

pub struct Scratch<'a> {
    pub precast: &'a dyn Fn(u32, u32) -> Option<Tensor>,

    pub partials: &'a dyn Fn(u32, u32) -> Option<Tensor>,

    /// The register tile's input table for this call's activation, when the
    /// caller keeps one across the matmuls that read the same activation (a
    /// layer's projections of one normed row): the room `precast` would mint
    /// for (`rows`, `contraction`), and whether it already holds the
    /// activation's rows with at least `counters` arrival counters zeroed, so
    /// the call skips its table pass. `None` lays the table every call.
    pub table: Option<KeptTable<'a>>,
}

pub fn matmul(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(
        ctx,
        tuning,
        "linear.matmul",
        act,
        w,
        y,
        scratch,
        capacity_rows,
    )
}

/// The leading `columns` columns of `act x w^T`, landed in place in `y`,
/// whose width is the row stride: the rest of each row is left alone, for
/// another engine to fill. `w` holds at least `columns` rows.
#[allow(clippy::too_many_arguments)]
pub fn matmul_columns(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Bank,
    y: Tensor,
    columns: u32,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt_over(
        ctx,
        tuning,
        "linear.matmul",
        act,
        w,
        y,
        Some(columns),
        scratch,
        capacity_rows,
    )
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(
        ctx,
        tuning,
        "linear.matmul",
        act,
        w,
        y,
        scratch,
        capacity_rows,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt_over(ctx, tuning, op, act, w, y, None, scratch, capacity_rows)
}

/// `act x w^T` over `y`'s leading `columns` (every column when `None`), at
/// `y`'s width as the row stride.
#[allow(clippy::too_many_arguments)]
fn act_x_wt_over(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    columns: Option<u32>,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    // PTQ1_0 is a single-plane inline-scale ternary bank: the fp16 scale lives
    // inside each 28-byte block, so there is no `scales`/`biases` plane to pair.
    // Route it to its own decode-in-dot kernel before the affine 3-plane checks.
    if w.codes.dtype == Dtype::Ptq1_0 {
        let (rows, stride, contraction) = extent(op, act, y)?;
        if columns.is_some_and(|c| c != stride) {
            return Err(refuse(op, "the ternary kernel lands whole rows only"));
        }
        if rows == 0 {
            return Ok(());
        }
        return qmv::ptq1_0::fire(ctx, op, act, w, y, rows, stride, contraction);
    }
    let Some(biases) = w.biases else {
        return Err(refuse(
            op,
            format!(
                "the weight is a symmetric {}-bit bank in groups of {}, and this plane \
                 stamps no dense point for one: `qmv.metal` instantiates the \
                 mxfp4 codec only at the routed shapes",
                w.bits, w.group
            ),
        ));
    };
    let (rows, stride, contraction) = extent(op, act, y)?;
    let columns = match columns {
        Some(c) if c == 0 || c > stride => {
            return Err(refuse(
                op,
                format!("{c} columns do not fit a row of {stride}"),
            ));
        }
        Some(c) => c,
        None => stride,
    };
    if rows == 0 {
        return Ok(());
    }
    if contraction % w.group != 0 {
        return Err(refuse(
            op,
            format!(
                "the contraction is {contraction}, not a whole number of {}-code \
                 groups: every point indexes its scales as `k / group`, so a \
                 partial group reads the next row's factor",
                w.group
            ),
        ));
    }
    let group = stated(op, w.group)?;
    let bits = stated(op, w.bits)?;
    // `n` is the row stride every kernel lands by; `cols` is how many of
    // the row this call computes, and sizes the grids.
    let (m, n, k) = (
        stated(op, rows)?,
        stated(op, stride)?,
        stated(op, contraction)?,
    );
    let cols = stated(op, columns)?;
    if w.ld < contraction || !w.ld.is_multiple_of(w.group) {
        return Err(refuse(
            op,
            format!(
                "the bank's row stride is {} columns, which holds no {contraction}-wide row \
                 of whole {}-code groups",
                w.ld, w.group
            ),
        ));
    }
    let ld = stated(op, w.ld)?;
    let fp16 = tuning.fp16_gemm_format(w.bits, w.group);
    let at = tuning.qmm_at(rows, columns, contraction, w.bits, w.group);
    // The profile's `kernel.qmm` rungs that fit this shape, in order: each
    // runs if it finds its scratch, else the next one does.
    for choice in tuning.qmm(&at) {
        let fired = match choice {
            Qmm::Mpp => mpp::fire(
                ctx,
                tuning,
                (act, w, biases, y),
                &scratch,
                (rows, columns, stride, contraction),
                capacity_rows,
                (k, n, ld),
            )?,
            Qmm::Staged => {
                staged::TILE.unsigned_abs() <= capacity_rows
                    && staged::fire(ctx, op, (act, w, biases, y), &scratch, (m, n, k, cols, ld))
                        .map(|()| true)?
            }
            Qmm::Register => sg::fire(
                ctx,
                tuning,
                (act, w, biases, y),
                &scratch,
                contraction,
                (m, n, k, cols, ld),
            )?,
            Qmm::Tiled { crossover_tg } => tiled::fire(
                ctx,
                tuning,
                op,
                (act, w, biases, y),
                &scratch,
                (crossover_tg, capacity_rows, fp16),
                (m, n, k, cols, ld, group, bits),
            )?,
            Qmm::Qmv {
                rows_max,
                packs,
                crossover_tg,
            } => {
                qmv::fire(
                    ctx,
                    op,
                    (act, w, biases, y),
                    (rows_max, packs, crossover_tg),
                    (m, n, k, cols, ld, group, bits),
                )?;
                true
            }
        };
        if fired {
            return Ok(());
        }
    }
    unreachable!("`kernel.qmm` ends on the vector kernel, which serves every shape")
}

fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}
