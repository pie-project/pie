//! The one error type the shell answers with.
//!
//! The C++ shell reports failure three ways depending on which lane wrote the
//! function: a `bool` with a `std::string* error` out-parameter, a `Pso` whose
//! `obj` is null, and a `PIE_STATUS_*` int. All three are the same fact, and
//! the first two lose it whenever a caller passes `nullptr` for the out-param
//! -- which is the default argument, so it is what most call sites do.
//!
//! One `Result` instead. The reason is not tidiness: `compile_pso` returning
//! an invalid `Pso` is a value a caller can ignore, and `Result` is one it
//! cannot.

use std::fmt;
use std::path::PathBuf;

/// The shell's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// What the Metal shell can fail at.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// No Metal device answered. On Apple silicon this means the process
    /// cannot reach the GPU at all, not that the machine has no GPU.
    NoDevice,

    /// A shader source file could not be read.
    ShaderRead {
        /// The path that failed, as written in the `#include` or by the caller.
        path: PathBuf,
        /// The underlying I/O failure.
        source: std::io::Error,
    },

    /// A `#include "..."` chain nested deeper than the shell will follow.
    ///
    /// Metal's runtime compiler resolves no includes of its own, so the shell
    /// splices them itself and needs its own cycle bound -- an include cycle
    /// is otherwise an infinite loop rather than a compiler diagnostic.
    IncludeTooDeep {
        /// The file at which the limit was reached.
        path: PathBuf,
        /// The limit that was exceeded.
        limit: usize,
    },

    /// A `#include "` opened a filename that no closing quote ended.
    UnterminatedInclude {
        /// The file containing the malformed directive.
        path: PathBuf,
        /// Byte offset of the directive within that file.
        offset: usize,
    },

    /// Metal rejected a shader source or a pipeline descriptor.
    Compile {
        /// The entry point being built, for the message.
        function: String,
        /// What Metal said.
        message: String,
    },

    /// A Metal object the shell needs could not be created.
    ///
    /// Kept distinct from [`Error::Compile`] because the remedies differ: a
    /// compile failure is in the shader text and a creation failure is
    /// generally a budget, a descriptor, or an OS version.
    Create {
        /// What was being created (`"MTL4CommandQueue"`, `"heap"`, ...).
        what: &'static str,
        /// What Metal said, when it said anything.
        message: String,
    },

    /// The placement heap has no room left for a request.
    ///
    /// Carries both numbers because "out of memory" without them cannot
    /// distinguish a budget that is too small from an allocator that is
    /// leaking, and those are the two things this failure is ever caused by.
    HeapExhausted {
        /// Bytes asked for, after alignment.
        requested: u64,
        /// Bytes left in the heap.
        available: u64,
        /// Total heap size, for context.
        capacity: u64,
    },

    /// A byte span left the buffer it was meant to stay inside.
    ///
    /// Not a Metal failure -- Metal would not have reported one. Shared
    /// storage rounds every allocation up, so the bytes just past a slot are
    /// mapped and writable and belong to whatever was placed next. This is
    /// the only thing between the two.
    OutOfRange {
        /// Which span was short (`"source"`, `"destination"`, `"region"`).
        what: &'static str,
        /// Where the span started.
        offset: u64,
        /// How long it was.
        bytes: u64,
        /// How long the region is.
        len: u64,
    },

    /// A model or allocation is larger than the device will hold resident.
    ///
    /// Worth its own variant because Metal does not report it: every buffer
    /// is created, every bind succeeds, and the failure surfaces much later
    /// as a command buffer returning `kIOGPUCommandBufferCallbackErrorOutOfMemory`
    /// from three levels down. Refusing up front is the only place the real
    /// numbers are still in hand.
    WorkingSetExceeded {
        /// Bytes the caller wants resident.
        requested: u64,
        /// What the device says it will hold.
        working_set: u64,
    },

    /// This fire cannot be RECORDED, and encoding it is correct.
    ///
    /// The only failure in this crate that a caller is meant to swallow, and
    /// it has its own variant so that swallowing it is a `match` rather than
    /// an `.ok()`.
    ///
    /// Recording is an optimisation: replaying an indirect command buffer
    /// costs 39.8 us where encoding the same 424 dispatches costs 14.87 ms.
    /// But an ICB binds BUFFERS, so it can only be made once every region a
    /// fire's operands point into has been registered — and a caller that
    /// has not registered its weights is not broken, it is un-optimised. The
    /// encode path binds addresses and does not care.
    ///
    /// **Everything else `record` can fail at is a bug.** A symbol with no
    /// compiled pipeline means the plan drifted from what was compiled; a
    /// dispatch that states scalars with nothing staged means the plan
    /// drifted from what was staged; a device that declines an ICB is a
    /// device failure. Under one variant and an `.ok()`, all four arrived as
    /// "fall back to encoding", so three real faults became a 374x
    /// regression and no message. That is the whole reason this variant
    /// exists: the fallback is now the narrow case, by name.
    Unrecordable {
        /// What could not be turned into a command, for the message.
        what: &'static str,
        /// Why not.
        message: String,
    },

    /// A launch program the interpreter cannot run.
    ///
    /// Distinct from every other variant because nothing here is the
    /// machine's fault: the device is fine, the memory is fine, and the text
    /// that arrived describes an operation, a shape, or a stage graph that
    /// does not make sense. Folding it into [`Error::Create`] would send a
    /// reader looking at budgets and descriptors for a fault whose fix is in
    /// the program.
    Program {
        /// What the interpreter could not make sense of.
        message: String,
    },

    /// A request this backend does not serve, for a reason the machine has
    /// nothing to do with.
    ///
    /// The seam's own refusals, and they are all one shape: a verb called
    /// before `load_model` allocated what it needs, a checkpoint whose family
    /// no Metal text states, a boot config with no `[model] descriptor`, a
    /// verb this backend does not implement. Nothing is exhausted, nothing
    /// failed to compile, and no device declined anything — the fix is in the
    /// deployment or in the call order, never in the machine.
    ///
    /// That is exactly the distinction [`Error::Create`] cannot make. A
    /// caller looking at "unknown model family" and "the heap declined a
    /// buffer" wants to do opposite things, and under one variant with 44
    /// `what` strings the only way to tell them apart is to match on prose.
    /// This is one of the four remediation-keyed variants
    /// `.wiki/driver/real-metal-north-star.md` §10 asks for, landed where the
    /// facade needed it.
    Unserved {
        /// Which verb refused, for the message.
        what: &'static str,
        /// What it will not serve, and what would change that.
        message: String,
    },
}

/// The channel plane's one failure, adopted rather than translated.
///
/// The `driver` crate carries a `Program` variant of its own — it is the only
/// thing that layer can fail at — and this shell's is the same fact. The
/// conversion exists so a `?` on a `pipeline::` result lands here without a
/// match at every call site, which is what makes the extraction of
/// the channel plane into its own crate invisible to the code that calls it.
impl From<driver::Error> for Error {
    fn from(error: driver::Error) -> Self {
        match error {
            driver::Error::Program { message } => Self::Program { message },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => f.write_str("no Metal device available"),
            Self::ShaderRead { path, source } => {
                write!(
                    f,
                    "cannot read shader source '{}': {source}",
                    path.display()
                )
            }
            Self::IncludeTooDeep { path, limit } => write!(
                f,
                "include nesting deeper than {limit} at '{}' (cycle?)",
                path.display()
            ),
            Self::UnterminatedInclude { path, offset } => write!(
                f,
                "unterminated #include at byte {offset} of '{}'",
                path.display()
            ),
            Self::Compile { function, message } => {
                write!(f, "compiling '{function}': {message}")
            }
            Self::Create { what, message } => write!(f, "creating {what}: {message}"),
            Self::HeapExhausted {
                requested,
                available,
                capacity,
            } => write!(
                f,
                "heap exhausted: {requested} bytes requested, {available} of {capacity} free"
            ),
            Self::OutOfRange {
                what,
                offset,
                bytes,
                len,
            } => write!(
                f,
                "{what} span of {bytes} bytes at offset {offset} leaves a region of {len} bytes"
            ),
            Self::Program { message } => {
                write!(f, "launch program cannot be interpreted: {message}")
            }
            Self::Unrecordable { what, message } => {
                write!(f, "{what} cannot be recorded, so it was encoded: {message}")
            }
            Self::Unserved { what, message } => {
                write!(f, "driver-metal: {what}: {message}")
            }
            Self::WorkingSetExceeded {
                requested,
                working_set,
            } => write!(
                f,
                "{requested} bytes exceeds the device working set of {working_set} bytes"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ShaderRead { source, .. } => Some(source),
            _ => None,
        }
    }
}
