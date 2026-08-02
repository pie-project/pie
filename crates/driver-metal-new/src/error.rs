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
