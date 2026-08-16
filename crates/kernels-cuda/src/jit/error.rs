//! Why a compile or a launch did not happen.
//!
//! # Three variants, and the three that went with the unit world
//!
//! This enum used to have six. `Unknown { symbol }` said *"no unit hosts the
//! symbol"*, `Missing { unit, symbol }` said *"the cubin loaded and the ROW's
//! entry is not in it"*, and `Geometry { symbol, why }` said *"the ROW's
//! launch rule could not be evaluated over these dims"*. All three name
//! something that no longer exists — a unit, a row, a `LaunchRule` the
//! runtime evaluates — and none of the three was ever constructed after the
//! per-symbol JIT landed: a routine body computes its own geometry as an
//! expression, and [`crate::routine`] answers "nothing declares this" as a
//! [`kernels::Refusal`] before a compile is ever asked for.
//!
//! They are deleted rather than kept for symmetry, because an unconstructible
//! variant of a public error type is a promise to a matcher that the promise
//! can never be kept to: a caller writing an arm for `Error::Geometry` is
//! writing dead code that reads as a handled case.

/// Why a compile or a launch did not happen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No CUDA device is current, so no architecture could be discovered.
    NoDevice,
    /// The root would not compile, or its instantiation is not in the image.
    Compile {
        /// The root that failed, by the name NVRTC calls it in diagnostics.
        unit: &'static str,
        /// What the compiler said.
        why: String,
    },
    /// The driver refused.
    Driver {
        /// The call that failed, spelled as CUDA spells it.
        what: &'static str,
        /// Its `CUresult`, as an integer.
        code: i32,
        /// Its `CUresult`, as a name.
        why: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => {
                write!(f, "no CUDA device is current, so no architecture could be discovered")
            }
            Self::Compile { unit, why } => write!(f, "`{unit}` would not compile: {why}"),
            Self::Driver { what, code, why } => write!(f, "{what} failed with {code} ({why})"),
        }
    }
}

/// No variant wraps another error, so there is no source to hand back.
///
/// [`Error::Compile`] and [`Error::Driver`] both carry a `String` that came
/// from NVRTC or from `CUresult`, and neither of those is a Rust error to
/// chain to. The impl exists so that a caller may treat this as a
/// `Box<dyn Error>`, which several of `driver-cuda`'s fire paths do.
impl std::error::Error for Error {}
