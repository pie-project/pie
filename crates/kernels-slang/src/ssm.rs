//! State-space and linear-attention mixers and their short convs, from
//! `kernels/ssm/`. Each walks a recurrent state pool: a bank of slots, one a
//! request, and the slot each row (or each request's first row) reads.

use dtype::Dtype;

use crate::encode::{Arg, ArgValue};
use crate::encode::{nonzero, refuse};
use crate::error::Error;
use crate::tensor::{RaggedTensor, SEGMENTS, Segments, Tensor, words_of};

pub mod conv1d;
pub mod gdn_prep;
pub mod gdn_recurrent;
pub mod kda;

/// A recurrent state pool: the bank of slots and the slot of each row.
/// `bank` carries the row's geometry (its width is one slot's elements);
/// the slots lie in the chunk pool `pool`, slot `s`'s row at word `origin +
/// s * slot_stride` (SPEC 10: `base + slot * stride + offset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatePool {
    pub bank: Tensor,

    pub slots: Tensor,

    pub pool: Segments,

    pub origin: u32,

    pub slot_stride: u32,
}

impl StatePool {
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
        }
    }

    /// The pool's four segment bindings, then its scalars (segment words,
    /// origin, slot stride): what every state kernel takes after its own
    /// buffers, its own scalars after.
    #[must_use]
    pub fn args(&self, mutable: bool) -> [ArgValue; SEGMENTS + 3] {
        let [a, b, c, d] = self.pool.args(mutable);
        [
            a,
            b,
            c,
            d,
            self.pool.words.arg(),
            self.origin.arg(),
            self.slot_stride.arg(),
        ]
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
