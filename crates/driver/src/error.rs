//! The one thing this layer can fail at.
//!
//! No device, heap or compiler diagnostic is reachable from here. What IS
//! reachable is a launch program that does not make sense -- an op the table
//! does not name, a stage graph closing over a value no stage produces, a
//! channel whose declared shape and dtype disagree with the cell it is bound
//! to -- so the one variant blames the *text*. The shells re-export their own
//! error and convert: `driver-metal` and `driver-cuda` each carry a `Program`
//! variant, so a `?` on one of this crate's results lands in the shell's type
//! without a match.

use std::fmt;

/// This layer's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// What the PTIR channel plane can fail at.
///
/// One variant, deliberately: a second would have to name something other
/// than the program. NOT `#[non_exhaustive]` -- a shell's `From` impl should
/// be a TOTAL match that breaks loudly when a variant is added.
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
