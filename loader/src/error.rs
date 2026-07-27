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

/// Every fallible step in the loader answers with this.
pub type Result<T> = std::result::Result<T, Error>;
