//! Why a compile or a launch did not happen.
//!
//! Three variants. `Unknown`, `Missing` and `Geometry` went with the unit
//! world: each named something that no longer exists — a unit, a row, a
//! `LaunchRule` the runtime evaluates — and none was constructible after the
//! per-symbol JIT landed, since a routine body computes its own geometry and
//! [`crate::routine`] answers "nothing declares this" as a
//! [`kernels::Refusal`] before a compile is asked for. They are deleted rather
//! than kept for symmetry: an unconstructible variant of a public error type
//! makes a caller's match arm dead code that reads as a handled case.

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

/// No variant wraps another error, so there is no source to hand back: both
/// carrying variants hold a `String` from NVRTC or `CUresult`, neither of
/// which is a Rust error to chain to. The impl exists so a caller may treat
/// this as a `Box<dyn Error>`, which several `driver-cuda` fire paths do.
impl std::error::Error for Error {}
