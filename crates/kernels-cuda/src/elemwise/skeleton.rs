//! The row skeleton (`elemwise/skeleton.cuh`): one row a block (or, narrower than
//! a warp's vectors, a run of a warp's lanes a row), held whole in
//! registers, around a body that computes it in hooks over the skeleton. The
//! body here is written by hand (`elemwise/skeleton.cu`), the skeleton's own
//! reference; generated ones ([`crate::synth`]) fire through the same
//! launch.

use super::BLOCK;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Kernel, Launch, aligned16, kernel, nonzero, refuse};
use crate::tensor::Tensor;
use dtype::Dtype;

const SKELETON: &str = "elemwise/skeleton.cu";

/// The elements of one vector a thread loads: a row's width is whole
/// vectors of it.
pub const VEC_WIDTH: u32 = 8;

/// The most vectors of 8 a thread holds of each plane: past it a row no
/// longer fits the registers.
pub const MAX_PER: u32 = 4;

/// The widest row the skeleton holds.
pub const MAX_WIDTH: u32 = BLOCK * VEC_WIDTH * MAX_PER;

/// `pie::elemwise::Plane`, field for field: a plane's first element, how
/// far apart its rows sit, in elements, the ids its rows are picked by
/// (0: each row its own), the vectors of 8 it repeats every (0: as wide as
/// the row), and its rows, the ids' bound (an id past them picks row 0, as
/// the gather kernels pick it).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    base: u64,
    row_stride: i64,
    ids: u64,
    period: u32,
    bound: u32,
}

impl Plane {
    /// A plane whose every row is its own.
    #[must_use]
    pub fn rows(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: i64::from(t.width),
            ids: 0,
            period: 0,
            bound: t.rows,
        }
    }

    /// One row every row reads (a norm's weight, its elements however `t`
    /// holds them); one narrower than the skeleton's rows is repeated across
    /// each (a head's weight, every head reading it).
    #[must_use]
    pub const fn shared(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: 0,
            ids: 0,
            period: t.rows.saturating_mul(t.width) / VEC_WIDTH,
            bound: 0,
        }
    }

    /// One element a row, its own (a position): `t`'s rows one element
    /// each, or the first of each.
    #[must_use]
    pub fn each(t: &Tensor) -> Self {
        Self::rows(t)
    }

    /// Row `r` reads the table's row `ids[r]` (an embedding's rows, a lane's
    /// modulation), `ids` a plane of i32 a row.
    #[must_use]
    pub fn gathered(table: &Tensor, ids: &Tensor) -> Self {
        Self {
            ids: ids.ptr,
            ..Self::rows(table)
        }
    }

    /// Columns `from..` of each of `t`'s rows: a row narrower than `t`'s
    /// read at an offset (half of a packed modulation, one layer's row of a
    /// stack).
    #[must_use]
    pub fn cols(t: &Tensor, from: u32) -> Self {
        Self {
            base: t.ptr + u64::from(from) * t.dtype.bytes_ceil(),
            ..Self::rows(t)
        }
    }

    /// The plane read through `ids` (an i32 a row): row `r` reads the plane's
    /// row `ids[r]`.
    #[must_use]
    pub const fn by(self, ids: &Tensor) -> Self {
        Self {
            ids: ids.ptr,
            ..self
        }
    }

    /// The plane's first element.
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }
}

/// How a skeleton's parameter is stored, two bits of `DT` each
/// (`pie::elemwise::kBf16`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stored {
    Bf16 = 0,
    F32 = 1,
    I32 = 2,
}

/// The threads a row of `width` takes: the fewest, a power of two, that
/// hold its vectors of 8 one each, at most a block of [`BLOCK`] (which then
/// holds up to [`MAX_PER`] each); several rows a block where they are fewer.
#[must_use]
pub fn lanes(width: u32) -> u32 {
    width
        .div_ceil(VEC_WIDTH)
        .max(1)
        .next_power_of_two()
        .min(BLOCK)
}

/// How a skeleton lays its rows over blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    /// Blocks of [`BLOCK`] threads, as many rows a block as it holds: a
    /// prefill's many rows, a block's table and barriers shared by them.
    Packed,
    /// A row a block of its own threads (a warp where a row is narrower,
    /// as many rows as it holds): a decode's few rows spread over the
    /// multiprocessors, each finishing on its own.
    Spread,
}

impl Schedule {
    pub const ALL: [Schedule; 2] = [Schedule::Packed, Schedule::Spread];

    /// The schedule for `rows` rows of `width` on a device of
    /// `multiprocessors`: spread until packing them would fill each
    /// multiprocessor's eight blocks (measured on an L40S: Gemma 4's heads
    /// and dots3's k row land faster spread through a prefill's 512 rows,
    /// packed at 4096), packed past it.
    #[must_use]
    pub fn of(width: u32, rows: u32, multiprocessors: u32) -> Schedule {
        let packed = rows.div_ceil(BLOCK / lanes(width));
        if block(width, Schedule::Spread) < BLOCK && packed < 8 * multiprocessors {
            Schedule::Spread
        } else {
            Schedule::Packed
        }
    }
}

/// The threads of a block of `width`-wide rows under `schedule`.
#[must_use]
pub fn block(width: u32, schedule: Schedule) -> u32 {
    match schedule {
        Schedule::Packed => BLOCK,
        Schedule::Spread => lanes(width).max(32),
    }
}

/// The vectors of 8 a thread holds of each plane of a `width`-wide row, or
/// why the skeleton does not hold such a row.
pub(crate) fn per(op: &'static str, width: u32) -> Result<u32, Error> {
    let width = nonzero(op, "the row's width", width)?;
    if !width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!("a {width}-wide row is not whole vectors of {VEC_WIDTH}"),
        ));
    }
    if width > MAX_WIDTH {
        return Err(refuse(
            op,
            format!("a {width}-wide row is wider than the skeleton holds ({MAX_WIDTH})"),
        ));
    }
    Ok((width / VEC_WIDTH).div_ceil(lanes(width)))
}

/// The floats of shared memory a skeleton of `width`-wide rows in blocks of
/// `block` threads whose body shares its row takes: a block's rows of f32.
#[must_use]
pub fn shared_floats(width: u32, block: u32) -> u32 {
    let lanes = lanes(width);
    block * (width / VEC_WIDTH).div_ceil(lanes) * VEC_WIDTH
}

/// The skeleton's kernel around `body` (a type the unit defines) for `ins`
/// parameters stored as `stored` and `outs` results, rows `width` wide in
/// blocks as `schedule` lays them, the parameters `ones` sets (by bit) read
/// as one element. A `read_only` skeleton's results land over none of its
/// parameters, so it reads them through the read-only path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn entrypoint(
    body: &str,
    width: u32,
    per: u32,
    stored: &[Stored],
    outs: usize,
    ones: u32,
    read_only: bool,
    schedule: Schedule,
) -> String {
    let dt = (0u32..)
        .zip(stored)
        .fold(0u64, |dt, (k, form)| dt | (*form as u64) << (2 * k));
    format!(
        "::pie::elemwise::row_skeleton<{body}, {}, {}, {per}, {}, {outs}, {ones}u, {dt}ull, {read_only}>",
        block(width, schedule),
        lanes(width),
        stored.len()
    )
}

/// The shared memory a body takes beside its skeleton: `table` floats of what
/// its columns alone compute, and (`shares`) its row's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Shared {
    pub table: u32,
    pub shares: bool,
}

/// Fires `kernel`, a row skeleton of [`entrypoint`] under `schedule` (`fields`
/// its body's members by value), over `rows` rows of planes, each row cut into
/// `parts` of the skeleton's `width` wide: `ins[k]` is the body's parameter
/// `k` (stored as `stored[k]`) and `outs[k]` its result `k`, a parameter
/// `ones` sets (by bit) one element (a row's, every part of it reading it),
/// read as is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fire(
    ctx: &Ctx,
    op: &'static str,
    kernel: &'static Kernel,
    fields: ArgValue,
    (ins, stored): (&[Plane], &[Stored]),
    outs: &[Plane],
    (rows, parts): (u32, u32),
    width: u32,
    ones: u32,
    shared: Shared,
    schedule: Schedule,
) -> Result<(), Error> {
    let rows = nonzero(op, "rows", rows)?;
    let parts = nonzero(op, "a row's parts", parts)?;
    let skeleton_rows = rows
        .checked_mul(parts)
        .ok_or_else(|| refuse(op, format!("{rows} rows of {parts} parts overflow a grid")))?;
    // a row plane's every row starts on 16 bytes: a whole number of 8 bf16
    // or 4 f32 apart
    let mut rowed = (0u32..)
        .zip(ins.iter().zip(stored))
        .filter(|(k, _)| (ones >> k) & 1 == 0)
        .map(|(_, (p, form))| (p, *form == Stored::F32))
        .chain(outs.iter().map(|p| (p, false)));
    if !rowed.all(|(p, wide)| aligned16(p.base) && p.row_stride % if wide { 4 } else { 8 } == 0) {
        return Err(refuse(op, "a plane is not 16-byte aligned at every row"));
    }
    // `pie::elemwise::Planes<IN, OUT>`: the planes, then the width and the
    // skeleton's rows, the parts and the table's floats, two to an 8-byte
    // word.
    let planes: Vec<u64> = ins
        .iter()
        .chain(outs)
        .flat_map(|p| {
            [
                p.base,
                p.row_stride.cast_unsigned(),
                p.ids,
                u64::from(p.period) | u64::from(p.bound) << 32,
            ]
        })
        .chain([
            u64::from(width) | u64::from(skeleton_rows) << 32,
            u64::from(parts) | u64::from(shared.table) << 32,
        ])
        .collect();
    let threads = block(width, schedule);
    let blocks = skeleton_rows.div_ceil(threads / lanes(width));
    let floats = shared.table
        + if shared.shares {
            shared_floats(width, threads)
        } else {
            0
        };
    ctx.fire(
        op,
        // The skeleton waits on its producer in-kernel (`kernel.pdl`).
        Fire::at(kernel).apply(
            Launch::per_row(blocks, threads)
                .smem(floats * 4)
                .programmatic(ctx.tuning().pdl),
        ),
        &[ArgValue::of_slice(&planes), fields, ctx.stage()],
    )
}

/// A residual norm as a row skeleton's body: `y += x` in place, then `out` the
/// RMS norm of `y` by `weight` (by `1 + weight` with `plus_one`), its
/// squares summed a pair at a time. Rows up to 8192 bf16 wide, every plane
/// 16-byte aligned.
pub fn residual_add_rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.residual_add_rmsnorm";
    if y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: y.dtype,
        });
    }
    debug_assert!(
        x.rows == y.rows && x.width == y.width && out.rows == y.rows && out.width == y.width,
        "the residual, the stream and the normed row share one shape"
    );
    let per = per(OP, y.width)?;
    let body = format!("::pie::elemwise::ResidualAddRmsnorm<{plus_one}>");
    fire(
        ctx,
        OP,
        kernel(
            SKELETON,
            &entrypoint(
                &body,
                y.width,
                per,
                &[Stored::Bf16; 3],
                2,
                0,
                false,
                Schedule::Packed,
            ),
        ),
        eps.arg(),
        (
            &[Plane::rows(&x), Plane::rows(y), Plane::shared(&weight)],
            &[Stored::Bf16; 3],
        ),
        &[Plane::rows(out), Plane::rows(y)],
        (y.rows, 1),
        y.width,
        0,
        Shared::default(),
        Schedule::Packed,
    )
}
