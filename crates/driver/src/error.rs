//! The one thing this layer can fail at.
//!
//! A device shell's error type is a list of the ways a machine can refuse:
//! no device, heap exhausted, working set exceeded, a compiler diagnostic.
//! None of those are reachable from here, and that is the point of this file
//! being three lines of substance instead of two hundred.
//!
//! What IS reachable is a launch program that does not make sense — an op the
//! table does not name, a stage graph that closes over a value no stage
//! produces, a channel whose declared shape and dtype disagree with the cell
//! it is bound to. Every one of those is the *text*'s fault, not the
//! machine's, and a caller that reads `Program` knows to look at the program.
//!
//! The shells re-export their own error and convert: `driver-metal` and
//! `driver-cuda` each carry a `Program` variant of their own, so a `?` on
//! one of this crate's results lands in the shell's type without a match.

use std::fmt;

/// This layer's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// What the PTIR channel plane can fail at.
///
/// One variant, deliberately. A second would have to name something other
/// than the program, and there is nothing else here to blame.
///
/// NOT `#[non_exhaustive]`, which is the reflex for a public error enum and
/// would be wrong here. The attribute buys the freedom to add a variant, and
/// it is paid for by every downstream shell writing a `_ =>` arm that can
/// never be taken — dead code a reader has to rule out. This enum's whole
/// claim is that there is exactly one thing a layer with no device can fail
/// at, so a shell's `From` impl should be a TOTAL match, and adding a variant
/// here should break it loudly rather than fall silently into a wildcard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A launch program the interpreter cannot run.
    Program {
        /// What could not be made sense of.
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program { message } => {
                write!(f, "launch program cannot be interpreted: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}
