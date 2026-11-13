//! State-space and linear-attention mixers and their short convs, from
//! `kernels/ssm/`. Each walks a recurrent state pool: a bank of slots, one a
//! request, and the slot each row (or each request's first row) reads.

use dtype::Dtype;

use crate::encode::{Arg, ArgValue};
use crate::encode::{nonzero, refuse};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Segments, Tensor, words_of};

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

fn requests(op: &'static str, x: RaggedTensor) -> Result<u32, Error> {
    csr_lanes(op, "query", x.indptr)
}

fn window_lanes(op: &'static str, indptr: Tensor) -> Result<u32, Error> {
    csr_lanes(op, "window", indptr)
}

/// The rows of a fire, at least one.
fn rows(op: &'static str, t: Tensor) -> Result<u32, Error> {
    nonzero(op, "rows", t.rows)
}
