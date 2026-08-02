//! What can go wrong, by whose fault it is.
//!
//! The compiler used to answer every question with `InvalidInput(String)` — two
//! variants and 284 construction sites, so the type carried no information and
//! the string carried all of it. A C caller could not match on it, a test could
//! not assert on it without matching prose, and nothing distinguished "your
//! contract asks for something impossible" from "this checkpoint is corrupt"
//! from "the compiler broke".
//!
//! Each variant below names a *place* the fault lives, which is the only
//! distinction a caller can act on:
//!
//! | Variant | Whose fault | What the caller should do |
//! | --- | --- | --- |
//! | [`Error::Contract`] | the driver's declaration | fix the contract |
//! | [`Error::Shard`] | the declaration, at this TP degree | fix the contract or the degree |
//! | [`Error::Checkpoint`] | the file on disk | get a different checkpoint |
//! | [`Error::Unsupported`] | the target's executor | use another backend, or teach it |
//! | [`Error::Overflow`] | a size no representation holds | none — the model is too big for this field |
//! | [`Error::Internal`] | the loader | file a bug |

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// The contract asks for something the algebra or the checkpoint cannot
    /// give: a missing name, a shape that disagrees with its expression, a
    /// duplicate declaration.
    #[error("contract: {0}")]
    Contract(String),

    /// The contract is well-formed but does not survive this tensor-parallel
    /// degree — an axis that does not divide, or a rank out of range.
    ///
    /// Separate from [`Error::Contract`] because it is the one failure that
    /// depends on a *target* number rather than on the declaration alone, and
    /// the driver's recovery is different: change `tp_size`, not the model.
    #[error("shard: {0}")]
    Shard(String),

    /// The checkpoint's own metadata is missing, malformed or inconsistent
    /// with itself. Nothing the driver declared caused this.
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

impl Error {
    /// The stable code this error crosses the C ABI as.
    ///
    /// A caller that cannot read the message can still tell a bad contract from
    /// a bad checkpoint, which is the whole reason the variants exist.
    pub const fn code(&self) -> u32 {
        match self {
            Self::Contract(_) => 1,
            Self::Shard(_) => 2,
            Self::Checkpoint(_) => 3,
            Self::Unsupported(_) => 4,
            Self::Overflow(_) => 5,
            Self::Internal(_) => 6,
        }
    }
}

/// zTensor's failures, mapped onto the loader's — in one place, because the
/// distinction is load-bearing and easy to lose.
///
/// `Unsupported` is the one that matters: a file that is perfectly valid but
/// uses a layout, encoding or capability this build does not implement is not
/// a malformed checkpoint, and it crosses the C ABI as a different
/// [`code`](Error::code). Flattening every zTensor error to `Checkpoint` would
/// tell a caller its file was broken when it was only unfamiliar.
impl From<ztensor::Error> for Error {
    fn from(err: ztensor::Error) -> Self {
        match err {
            ztensor::Error::Unsupported(_) => Self::Unsupported(err.to_string()),
            // A rejected file, a name that is not there, bad input on the
            // write path, an I/O failure while reading a header — from the
            // loader's side these are all "the checkpoint did not deliver".
            _ => Self::Checkpoint(err.to_string()),
        }
    }
}

#[cfg(test)]
mod ztensor_conversion {
    use super::*;

    /// The two zTensor outcomes a caller must be able to tell apart across the
    /// C ABI, where only the code survives.
    #[test]
    fn unsupported_survives_as_unsupported() {
        let unfamiliar = ztensor::Error::Unsupported("layout \"x.future/1\"".into());
        assert_eq!(Error::from(unfamiliar).code(), 4);

        let broken = ztensor::Error::NotFound("tensor \"w\"".into());
        assert_eq!(Error::from(broken).code(), 3);
    }
}

/// Every fallible step in the loader answers with this.
pub type Result<T> = std::result::Result<T, Error>;

/// Name what overflowed, without spelling out the `ok_or_else` each time.
///
/// The loader's arithmetic is all checked, which is right, but written out in
/// full it reads as ceremony rather than as the calculation: three lines of
/// `.checked_add(b).ok_or_else(|| Error::Overflow("…".to_string()))?` for one
/// addition. `plan/passes/memory.rs` was a fifth error handling by line count.
///
/// The `checked_*` call stays visible on purpose — this shortens the failure
/// arm, not the operation:
///
/// ```ignore
/// let end = offset.checked_add(bytes).or_overflow("persistent byte overflow")?;
/// ```
pub trait OrOverflow<T> {
    /// The message is passed through verbatim, so a site reads the same as the
    /// `Error::Overflow` it replaced.
    fn or_overflow(self, message: impl Into<String>) -> Result<T>;
}

impl<T> OrOverflow<T> for Option<T> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::Overflow(message.into()))
    }
}

/// Narrowing conversions answer with [`std::num::TryFromIntError`], not `None`.
///
/// Deliberately not implemented for every `Result`: [`Result<T>`] is the
/// loader's own, and letting `or_overflow` swallow one would relabel a contract
/// error as an overflow.
impl<T> OrOverflow<T> for std::result::Result<T, std::num::TryFromIntError> {
    fn or_overflow(self, message: impl Into<String>) -> Result<T> {
        self.map_err(|_| Error::Overflow(message.into()))
    }
}
