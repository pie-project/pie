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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    base: u64,
    row_stride: i64,
    ids: u64,
    period: u32,
    bound: u32,
    /// Where a result's row lands, past its plane's own rows: `pie::elemwise::Store`.
    store: Store,
}

/// `pie::elemwise::Store`: a result's vectors of 8 from `from` up to `to`
/// (every one where `to` is 0) landing from its plane row's first element;
/// or, paged, at the cache row `page[r] · page_size + slot[r]` (none where
/// the slot lies past the page or `valid[r]` is 0), as e4m3 elements
/// `e4m3(x / scale)` where `e4m3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Store {
    page: u64,
    slot: u64,
    valid: u64,
    from: u32,
    to: u32,
    page_size: u32,
    e4m3: bool,
    scale: f32,
    /// A head-major window's head and stride, in vectors (0: whole).
    head: u32,
    stride: u32,
    /// A block-scaled row's layout, packed ([`blocks_word`]), its floor
    /// `scale`; 0 where the result is not one.
    blocks: u32,
}

impl Store {
    const WHOLE: Store = Store {
        page: 0,
        slot: 0,
        valid: 0,
        from: 0,
        to: 0,
        page_size: 0,
        e4m3: false,
        scale: 1.0,
        head: 0,
        stride: 0,
        blocks: 0,
    };

    /// Its words, as `fire` passes them.
    fn words(self) -> [u64; 7] {
        [
            self.page,
            self.slot,
            self.valid,
            u64::from(self.from) | u64::from(self.to) << 32,
            u64::from(self.page_size) | u64::from(u32::from(self.e4m3)) << 32,
            u64::from(self.scale.to_bits()) | u64::from(self.head) << 32,
            u64::from(self.stride) | u64::from(self.blocks) << 32,
        ]
    }
}

/// A paged cache's plane as a result lands in it: its rows `cells` of
/// `width`, each row's page and slot (`page`, `slot`, i32 a row), which rows
/// are live (`valid`, a byte a row, or 0: every one), the page's rows, and
/// whether its elements are e4m3 under `scale` (else bf16).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cache {
    pub base: u64,
    pub width: u32,
    pub page: u64,
    pub slot: u64,
    pub valid: u64,
    pub page_size: u32,
    pub e4m3: Option<f32>,
    /// A block-scaled row's layout, where each row is one (its data rows,
    /// then its scale rows, a page).
    pub blocks: Option<kernels_ir::BlockRows>,
}

/// `rows` packed as `pie::elemwise::Store::blocks` reads it, or `None`
/// where a field does not fit its bits.
#[must_use]
pub fn blocks_word(rows: &kernels_ir::BlockRows) -> Option<u32> {
    use kernels_ir::{BlockScale, Code};
    let fits = |v: u32, bits: u32| v < 1 << bits;
    let (coded, tail, scales) = (rows.coded / 8, rows.tail / 8, rows.scale_bytes / 4);
    // a unit scale is e4m3 under a scale of one, the plain e4m3 store's
    (rows.verify()
        && rows.scale != BlockScale::Unit
        && rows.coded.is_multiple_of(8)
        && rows.tail.is_multiple_of(8)
        && rows.scale_bytes.is_multiple_of(4)
        && rows.block.is_multiple_of(8)
        // a row's blocks' maxima, at most `kBlocks` of them a row
        && rows.scales() <= 64
        && fits(rows.block, 8)
        && fits(coded, 8)
        && fits(tail, 8)
        && fits(scales, 5))
    .then(|| {
        1 | u32::from(rows.code == Code::E2m1) << 1
            | u32::from(rows.scale == BlockScale::E4m3) << 2
            | rows.block << 3
            | coded << 11
            | tail << 19
            | scales << 27
    })
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
            store: Store::WHOLE,
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
            store: Store::WHOLE,
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

    /// A result landing the columns `from..from + width` of each of its
    /// rows (a split's piece) in `t`, its rows `width` wide.
    #[must_use]
    pub fn window(t: &Tensor, from: u32, width: u32) -> Self {
        Self {
            store: Store {
                from: from / VEC_WIDTH,
                to: (from + width) / VEC_WIDTH,
                ..Store::WHOLE
            },
            ..Self::rows(t)
        }
    }

    /// A result landing `head` columns of every `stride` from `from` of
    /// each of its rows (a head-major slice: Qwen 3.5's q, interleaved with
    /// its gate a head), `width` of them in `t`, one head after another.
    #[must_use]
    pub fn window_strided(t: &Tensor, from: u32, width: u32, head: u32, stride: u32) -> Self {
        let heads = width / head.max(1);
        Self {
            store: Store {
                from: from / VEC_WIDTH,
                to: (from + heads.saturating_sub(1) * stride + head) / VEC_WIDTH,
                head: head / VEC_WIDTH,
                stride: stride / VEC_WIDTH,
                ..Store::WHOLE
            },
            ..Self::rows(t)
        }
    }

    /// A result landing the columns `from..from + width` of each row in
    /// `cache`'s row its page and slot name.
    #[must_use]
    pub fn paged(cache: Cache, from: u32, width: u32) -> Self {
        Self {
            base: cache.base,
            row_stride: i64::from(cache.width),
            ids: 0,
            period: 0,
            bound: 0,
            store: Store {
                page: cache.page,
                slot: cache.slot,
                valid: cache.valid,
                from: from / VEC_WIDTH,
                to: (from + width) / VEC_WIDTH,
                page_size: cache.page_size,
                e4m3: cache.e4m3.is_some(),
                scale: cache
                    .blocks
                    .map_or(cache.e4m3.unwrap_or(1.0), |rows| rows.floor),
                head: 0,
                stride: 0,
                blocks: cache.blocks.as_ref().and_then(blocks_word).unwrap_or(0),
            },
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
    /// multiprocessors, each finishing on its own. Every parameter is
    /// loaded up front, where packed rows issue a later pass's own late.
    Spread,
}

impl Schedule {
    pub const ALL: [Schedule; 2] = [Schedule::Packed, Schedule::Spread];

    /// The schedule for `rows` rows of `width` on a device of
    /// `multiprocessors`: rows narrower than a block are spread while
    /// packing them takes at most a block for each eight multiprocessors,
    /// packed past it (measured on an L40S, its packed rows issuing a later
    /// pass's parameters late: dots3's k row lands faster spread through 512
    /// rows, sixteen blocks packed; Gemma 4's and DeepSeek-V4's heads and
    /// Gemma 4's per-layer rows packed from 32 blocks). A row a block either way (a row as
    /// wide as a block) is spread while each multiprocessor holds at most
    /// two: its parameters all in flight at once, as a decode's latency
    /// wants, where more rows want the registers a later pass's parameters
    /// hold until then (a DiT block's modulation through its norm, at 512
    /// rows).
    #[must_use]
    pub fn of(width: u32, rows: u32, multiprocessors: u32) -> Schedule {
        let packed = rows.div_ceil(BLOCK / lanes(width));
        let spread = if block(width, Schedule::Spread) < BLOCK {
            8 * packed <= multiprocessors
        } else {
            rows <= 2 * multiprocessors
        };
        if spread {
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
/// as one element, those `late` sets loaded where the body asks
/// (`load<k>()`) and those `stream` sets fetched a vector at a time where
/// each pass reaches it (`fetch<m>(p)`), never held across passes, those
/// `vectors` sets read as small vectors one a row (`f.vec`), and with
/// `side` side blocks beside the rows' ([`fire`] two blocks high). A
/// `read_only` skeleton's results land over none of its parameters, so it
/// reads them through the read-only path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn entrypoint(
    body: &str,
    width: u32,
    per: u32,
    stored: &[Stored],
    outs: usize,
    (ones, late, stream, vectors, side): (u32, u32, u32, u32, bool),
    read_only: bool,
    schedule: Schedule,
) -> String {
    let dt = (0u32..)
        .zip(stored)
        .fold(0u64, |dt, (k, form)| dt | (*form as u64) << (2 * k));
    format!(
        "::pie::elemwise::row_skeleton<{body}, {}, {}, {per}, {}, {outs}, {ones}u, {dt}ull, {read_only}, {late}u, {stream}u, {vectors}u, {side}>",
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
/// read as is; a kernel of `side` blocks two blocks high.
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
    (ones, vectors, vector_outs, side): (u32, u32, u32, bool),
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
    // (a small vector one a row is read and landed an f32 at a time)
    let mut rowed = (0u32..)
        .zip(ins.iter().zip(stored))
        .filter(|(k, _)| ((ones | vectors) >> k) & 1 == 0)
        .map(|(_, (p, form))| (p, *form == Stored::F32))
        .chain(
            (0u32..)
                .zip(outs)
                .filter(|(k, _)| (vector_outs >> k) & 1 == 0)
                .map(|(_, p)| (p, false)),
        );
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
        .chain(outs.iter().flat_map(|p| p.store.words()))
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
            Launch::grid([blocks, if side { 2 } else { 1 }, 1], [threads, 1, 1])
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
                (0, 0, 0, 0, false),
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
        (0, 0, 0, false),
        Shared::default(),
        Schedule::Packed,
    )
}
