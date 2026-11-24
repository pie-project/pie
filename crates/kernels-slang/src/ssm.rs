//! State-space and linear-attention mixers and their short convs, from
//! `kernels/ssm/`. Each walks a recurrent state pool: a bank of slots, one a
//! request, and the slot each row (or each request's first row) reads.

use dtype::Dtype;

use crate::encode::{Arg, ArgValue};
use crate::encode::{nonzero, refuse};
use crate::error::Error;
use crate::tensor::{Segments, Tensor, words_of};

pub mod conv1d;
pub mod gdn_prep;
pub mod gdn_recurrent;
pub mod kda;

/// A recurrent state pool: the bank of slots and the slots of each row.
/// `bank` carries the row's geometry (its width is one record's elements);
/// the slots lie in the chunk pool `pool`, slot unit `s`'s row at word
/// `origin + s * slot_stride` (SPEC 10: `base + slot * stride + offset`).
/// A record is `pieces` slots, `slots` holding that many units an entry;
/// this row's atoms (heads, channels) are cut `piece_atoms` a piece, piece
/// `p` in the entry's slot `piece_first + p` (`kernels/prim/pool.slang`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatePool {
    pub bank: Tensor,

    pub slots: Tensor,

    pub pool: Segments,

    pub origin: u32,

    pub slot_stride: u32,

    pub pieces: u32,

    pub piece_first: u32,

    pub piece_atoms: u32,

    /// Where a prompt's scans land the state as of rows inside a lane
    /// (CONTRACT-BC 16), when a lane of the fire snapshots.
    pub snapshots: Option<Snapshots>,
}

/// The snapshot table a fire's chunked and committed scans read
/// (`kernels/prim/pool.slang`, `PIE_STATE_SNAPSHOTS`): a lane,
/// [`RS_SNAPSHOTS`] rows into it (ascending, a row `<= 0` ends them), then
/// each snapshot's record as the pool's `pieces` units; request `r` of the
/// scan's CSR is lane `lane0 + r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Snapshots {
    pub table: Tensor,
    pub lane0: u32,
}

/// The rows inside one lane a fire lands the state of at most.
pub const RS_SNAPSHOTS: u32 = 4;

impl StatePool {
    /// `entry`'s twin that lands snapshots, where this fire has a table.
    pub(crate) fn entry(
        &self,
        op: &'static str,
        entry: &'static str,
    ) -> Result<&'static str, Error> {
        if self.snapshots.is_none() {
            return Ok(entry);
        }
        let twin = entry.replacen("_chunked", "_chunked_snap", 1).replacen(
            "_committed",
            "_committed_snap",
            1,
        );
        crate::encode::instantiated(op, twin)
    }

    /// The snapshot table's binding, before the pool's segments; none
    /// without a table.
    pub(crate) fn snap_table(&self) -> Vec<ArgValue> {
        self.snapshots
            .map(|s| vec![s.table.arg()])
            .unwrap_or_default()
    }

    /// The table's first lane, after a scan's scalars; none without one.
    pub(crate) fn snap_lane0(&self, op: &'static str) -> Result<Vec<ArgValue>, Error> {
        self.snapshots
            .map(|s| Ok(vec![crate::encode::stated(op, s.lane0)?.arg()]))
            .unwrap_or_else(|| Ok(Vec::new()))
    }

    /// A bank of its own: `bank`'s rows are the slots, one segment.
    #[must_use]
    pub fn bank(bank: Tensor, slots: Tensor) -> Self {
        let row = u64::from(bank.width) * bank.dtype.bits() / 32;
        Self {
            bank,
            slots,
            pool: Segments::one(bank.buf, words_of(bank).max(1)),
            origin: 0,
            slot_stride: u32::try_from(row).unwrap_or(u32::MAX),
            pieces: 1,
            piece_first: 0,
            piece_atoms: u32::MAX,
            snapshots: None,
        }
    }

    /// The pool's four segment bindings, then its scalars (segment words,
    /// origin, slot stride): what a state kernel takes after its own
    /// buffers; [`StatePool::piece_args`] adds the pieces.
    #[must_use]
    pub fn args(&self, mutable: bool) -> Vec<ArgValue> {
        let mut args = self.pool.args(mutable).to_vec();
        args.extend([
            self.pool.words.arg(),
            self.origin.arg(),
            self.slot_stride.arg(),
        ]);
        args
    }

    /// [`StatePool::args`], then the pieces' scalars.
    #[must_use]
    pub fn piece_args(&self, mutable: bool) -> Vec<ArgValue> {
        let mut args = self.args(mutable);
        args.extend([
            self.pieces.arg(),
            self.piece_first.arg(),
            self.piece_atoms.arg(),
        ]);
        args
    }
}

/// A speculative window's bookkeeping, a lane a request from `lane0`: the
/// rows replayed before each request's own, the row each request's slot is
/// kept as of, and each request's i32 slot (negative: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Committed {
    pub replay: Tensor,

    pub commit: Tensor,

    pub slots: Tensor,

    pub lane0: u32,
}

/// A buffering fire's staged rs table (`kernels/prim/carry.slang`), which
/// its recurrent kernels read each lane's buffered rows through where they
/// lie in the pool: lane `l` of the fire has [`CARRY_HEADER`] ints from int
/// `l * CARRY_HEADER` (the buffer token its first own row lands at, or -1;
/// the rows it carries ahead of its own; where its carried rows' tokens and
/// its buffer pages start in the table; the rows of its run its state is
/// committed at), its tokens and pages after. `slots` is the fire's lanes'
/// records ([`Committed::slots`]); the window's lane `r` is the fire's
/// `lane0 + r`. An rs page is a slot of the pool's rs list, `stride_words`
/// words of the pool, `page_tokens` tokens of every plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Carry {
    pub table: Tensor,
    pub slots: Tensor,
    pub lane0: u32,
    pub page_tokens: u32,
    pub stride_words: u32,
}

/// The ints a lane's header takes in a [`Carry`] table.
pub const CARRY_HEADER: u32 = 5;

/// Where a buffered plane lies in an rs page: the word it starts at, and
/// the words a token's row of it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CarryPlane {
    pub at: u32,
    pub row: u32,
}

impl Carry {
    /// The pages' scalars, refused where a page holds no token.
    pub(crate) fn page_args(&self, op: &'static str) -> Result<[ArgValue; 2], Error> {
        Ok([
            nonzero(op, "the tokens an rs page holds", self.page_tokens)?.arg(),
            self.stride_words.arg(),
        ])
    }
}

impl CarryPlane {
    pub(crate) fn args(&self) -> [ArgValue; 2] {
        [self.at.arg(), self.row.arg()]
    }
}

/// The widest head a scan stages in workgroup memory.
const HEAD_MAX: u32 = 256;

fn head_width(op: &'static str, width: u32, what: &'static str) -> Result<(), Error> {
    if width > HEAD_MAX {
        return Err(refuse(
            op,
            format!("{what} is {width}, above the {HEAD_MAX}-wide row this scan stages"),
        ));
    }
    Ok(())
}

/// The requests an i32 CSR spans, at least one.
fn csr_lanes(op: &'static str, what: &str, indptr: Tensor) -> Result<u32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the {what} CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                indptr.dtype
            ),
        ));
    }
    match indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(
            op,
            format!("the {what} CSR this fire names spans no request"),
        )),
    }
}

/// Refuses a committed dispatch of more lanes than its work plane holds:
/// a fire's lanes take as many dispatches as the plane holds lanes.
fn work_holds(op: &'static str, work: Tensor, csr: Csr) -> Result<(), Error> {
    if csr.lanes > work.rows {
        return Err(refuse(
            op,
            format!(
                "a dispatch of {} lanes, and the work plane holds {}",
                csr.lanes, work.rows
            ),
        ));
    }
    Ok(())
}

/// The rows a window's dispatch walks of the fire's whole row tensors: from
/// fire row `row0`, `rows` of them. The tensors are bound whole, one view
/// whatever window a fire cuts; the kernels add `row0` to each row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowSpan {
    pub row0: u32,
    pub rows: u32,
}

impl RowSpan {
    /// Every row of `t`, from row 0.
    #[must_use]
    pub fn whole(t: Tensor) -> RowSpan {
        RowSpan {
            row0: 0,
            rows: t.rows,
        }
    }

    /// The span's rows `t` (bound whole) holds, as a cut of it from the
    /// span's first row would: a value's room may hold fewer than the
    /// window. Refused where it holds none.
    pub(crate) fn held(&self, op: &'static str, t: Tensor) -> Result<u32, Error> {
        nonzero(op, "rows", self.rows.min(t.rows.saturating_sub(self.row0)))
    }

    /// The span's rows, refused where they pass any of `held` (tensors
    /// whose rows it indexes) or are none.
    pub(crate) fn rows(&self, op: &'static str, held: &[Tensor]) -> Result<u32, Error> {
        let rows = nonzero(op, "rows", self.rows)?;
        let end = u64::from(self.row0) + u64::from(rows);
        for t in held {
            if end > u64::from(t.rows) {
                return Err(refuse(
                    op,
                    format!(
                        "the window's rows {}..{end} pass the {} rows bound",
                        self.row0, t.rows
                    ),
                ));
            }
        }
        Ok(rows)
    }
}

/// A window's query CSR in the fire's packed windows table: the table is
/// bound whole (one view whatever window a fire cuts) and the window's
/// `lanes + 1` boundaries lie from int `at` of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Csr {
    pub table: Tensor,
    pub at: u32,
    pub lanes: u32,
}

impl Csr {
    /// The lanes the CSR spans, at least one, refused where its boundaries
    /// pass the table or are not i32.
    fn lanes(&self, op: &'static str) -> Result<u32, Error> {
        let lanes = csr_lanes(
            op,
            "window",
            Tensor {
                rows: self.lanes + 1,
                ..self.table
            },
        )?;
        let held = u64::from(self.table.rows) * u64::from(self.table.width);
        if u64::from(self.at) + u64::from(lanes) + 1 > held {
            return Err(refuse(
                op,
                format!(
                    "the window's {} boundaries from int {} pass the {held} ints of its table",
                    lanes + 1,
                    self.at
                ),
            ));
        }
        Ok(lanes)
    }
}

/// The rows of a fire, at least one.
fn rows(op: &'static str, t: Tensor) -> Result<u32, Error> {
    nonzero(op, "rows", t.rows)
}
