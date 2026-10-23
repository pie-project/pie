//! State-space and linear-attention mixers and their short convs, from
//! `kernels/ssm/`. Each walks a recurrent state pool: a bank of slots, one a
//! request, and the slot each row (or each request's first row) reads.

use dtype::Dtype;

use crate::encode::{nonzero, refuse};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

pub mod conv1d;
pub mod gdn_prep;
pub mod gdn_recurrent;
pub mod kda;

/// A recurrent state pool: the f32 bank of slots and the u32 slot of each
/// row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatePool {
    pub bank: Tensor,

    pub slots: Tensor,
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
