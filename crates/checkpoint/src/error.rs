//! What can go wrong, by whose fault it is.
//!
//! Each variant names a *place* the fault lives, which is the only
//! distinction a caller can act on:
//!
//! | Variant | Whose fault | What the caller should do |
//! | --- | --- | --- |
//! | [`Error::Contract`] | the engine's declaration | fix the contract |
//! | [`Error::Shard`] | the declaration, at this TP degree | fix the contract or the degree |
//! | [`Error::Checkpoint`] | the file on disk | get a different checkpoint |
//! | [`Error::Unsupported`] | the target's executor | use another backend, or teach it |
//! | [`Error::Overflow`] | a size no representation holds | none — the model is too big for this field |
//! | [`Error::Internal`] | the loader | file a bug |

use thiserror::Error;

/// What the loader refuses on, one variant per recovery the table above names.
#[derive(Debug, Error)]
pub enum Error {
    /// The contract asks for something the algebra or the checkpoint cannot
    /// give: a missing name, a shape that disagrees with its expression, a
    /// duplicate declaration.
    #[error("contract: {0}")]
    Contract(String),

    /// Well-formed but does not survive this tensor-parallel degree (an axis
    /// that does not divide, or a rank out of range); separate from
    /// [`Error::Contract`] since the recovery differs: change `tp_size`.
    #[error("shard: {0}")]
    Shard(String),

    /// The checkpoint's own metadata is missing, malformed or inconsistent
    /// with itself. Nothing the engine declared caused this.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// The plan is well-formed, but this target's executor has no instruction
    /// for it.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A size, offset or stride left the range its representation holds.
    #[error("overflow: {0}")]
    Overflow(String),

    /// A compiler invariant broke. Not reachable from any contract.
    #[error("internal: {0}")]
    Internal(String),
}

/// zTensor's failures, mapped onto the loader's. `Unsupported` is the one
/// that matters: a layout this build does not implement is not a malformed
/// checkpoint, and a caller retrying against a newer build must be able to
/// tell the two apart. A container version this build does not implement is
/// the same kind of fact (zTensor states it as `Reject { rule:
/// Rule::Version }`): the bytes are faithful, just from a newer or older
/// pie, and the recovery is to re-import, not find a different file.
impl From<ztensor::Error> for Error {
    fn from(err: ztensor::Error) -> Self {
        match err {
            ztensor::Error::Unsupported(_) => Self::Unsupported(err.to_string()),
            // Vocabulary this build does not implement, not bytes that
            // failed to deliver: recovery is a different build.
            ztensor::Error::Reject {
                rule: ztensor::Rule::Version,
                ..
            } => Self::Unsupported(err.to_string()),
            // A rejected file, a name that is not there, bad input on the
            // write path, an I/O failure while reading a header — from the
            // loader's side these are all "the checkpoint did not deliver".
            _ => Self::Checkpoint(err.to_string()),
        }
    }
}

/// Every fallible step in the loader answers with this.
pub type Result<T> = std::result::Result<T, Error>;

/// Name what overflowed, without spelling out `ok_or_else` each time. The
/// `checked_*` call stays visible; this only shortens the failure arm:
///
/// ```ignore
/// let end = offset.checked_add(bytes).or_overflow("persistent byte overflow")?;
/// ```
pub trait OrOverflow<T> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T>;
}

impl<T> OrOverflow<T> for Option<T> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::Overflow(message.into()))
    }
}

/// Narrowing conversions answer with [`std::num::TryFromIntError`], not
/// `None`. Not implemented for every `Result`: that would let `or_overflow`
/// relabel a contract error as an overflow.
impl<T> OrOverflow<T> for std::result::Result<T, std::num::TryFromIntError> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.map_err(|_| Error::Overflow(message.into()))
    }
}

