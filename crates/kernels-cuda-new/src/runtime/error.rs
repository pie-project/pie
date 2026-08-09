//! Every way a fire can be refused, in one vocabulary.
//!
//! # Why one enum, where the ahead-of-time driver had three
//!
//! `driver-cuda` splits this three ways — `bind::device::Error` for a module
//! that will not load or a kernel that will not launch,
//! `bind::nvrtc::CompileError` for a source NVRTC rejects, and
//! `bind::nvrtc::FamilyError` for the pair of them — and that was the right
//! shape there, because each stage had a CALLER of its own. A test compiled a
//! unit and read a `CompileError`; a loader loaded an image and read an
//! `Error`; the two met only in `Unit::load`, which is why `FamilyError`
//! exists at all and why it has exactly two variants.
//!
//! Under the JIT there is one caller. [`fire`](fn@crate::runtime::fire)
//! discovers the architecture, compiles the unit, loads the module, resolves
//! the entry, evaluates the launch rule and binds the arguments — and every
//! one of those is a refusal it has to report to a dispatcher that knows about
//! none of them. Three enums would have to be joined by a fourth to say that,
//! and the fourth would be a type whose only content is which of the three it
//! wraps. So the join is the type, and the stages are its variants.
//!
//! [`crate::runtime::nvrtc::CompileError`] survives as its own type all the
//! same, because compiling is still a thing done on purpose: a test that hands
//! [`crate::runtime::nvrtc::compile_with`] a doctored header set wants the
//! compiler's diagnosis — which name would not resolve, which include did not
//! land — and not a launch's summary of it. [`Error::Compile`] is what a FIRE
//! says about that same failure, which is a different sentence for a different
//! reader.
//!
//! # `Unknown` is the variant that earns the `Result`
//!
//! A caller must be able to tell *"not mine"* from *"mine and broken"*. The
//! first means try another dispatcher; the second means stop, because a row
//! that reaches this crate has no shim entry left and there is nothing to fall
//! back to. The ahead-of-time `bind::jit::fire` drew that distinction with a
//! `bool` — `false` for "no unit hosts this", `true` for everything else,
//! including every failure — and so the only thing a caller could learn was
//! whether to keep looking. The reason went to a log and nowhere else, which
//! is how a driver ends up reporting `UnknownKernel` for a kernel it knows
//! about perfectly well and could not compile.
//!
//! A `Result` says both: [`Error::Unknown`] is "not mine", and every other
//! variant is "mine, and here is what happened".

/// Why a fire did not happen.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// No unit hosts the symbol.
    ///
    /// The only variant that means *try somewhere else*. Every other one is
    /// a unit claiming the row and failing it.
    Unknown {
        /// The symbol that was asked for. Owned, because it came from a
        /// dispatcher as a `&str` off a trace and there is no table entry to
        /// borrow it from — that is precisely what went wrong.
        symbol: String,
    },
    /// No CUDA device is current, so no architecture could be discovered.
    ///
    /// A cubin is per-`sm_XY` and [`crate::runtime::cache::arch`] asks the
    /// bound device what that is. A process that never bound one cannot
    /// compile, and saying so here is better than compiling for a guess.
    NoDevice,
    /// The unit would not compile.
    ///
    /// `why` is a rendered [`crate::runtime::nvrtc::CompileError`] and the
    /// NVRTC log is inside it. Flattened to a `String` rather than nested,
    /// because this error crosses a feature boundary into a dispatcher that
    /// has no reason to know NVRTC's vocabulary — and because the log is the
    /// whole of what a human needs from it.
    Compile {
        /// The unit that failed, by the name NVRTC calls it in diagnostics.
        unit: &'static str,
        /// What the compiler said.
        why: String,
    },
    /// The cubin loaded but the row's entry is not in it.
    ///
    /// Drift between the rows and the templates that survived the compile:
    /// the instantiation was named, NVRTC accepted the expression, and
    /// `cuModuleGetFunction` found nothing under the lowered name. Reported
    /// at load rather than at launch, because a driver that starts cleanly
    /// and dies on the first fire that needs the missing kernel is the same
    /// bug discovered later and by a worse reader.
    Missing {
        /// The unit whose image was searched.
        unit: &'static str,
        /// The row that named the entry.
        symbol: &'static str,
    },
    /// The row's launch rule could not be evaluated over these dims.
    ///
    /// Two of [`crate::runtime::Ungeometric`]'s three variants are drift and
    /// the third is a rectangle that collapsed; none is a condition to
    /// recover from, and all three are refused rather than clamped, because
    /// a zero grid launches nothing and reports success.
    Geometry {
        /// The row whose rule was evaluated.
        symbol: &'static str,
        /// Which way it did not work out.
        why: crate::runtime::Ungeometric,
    },
    /// The values did not match the row.
    ///
    /// Kept as [`crate::runtime::ArgError`] rather than flattened, because
    /// every variant of it names an operand INDEX and a type, and a caller
    /// that emits argument lists — which is the only kind of caller that can
    /// produce one — wants to match on that rather than to read it.
    Args(crate::runtime::ArgError),
    /// The row's arm could not be chosen or could not be filled.
    ///
    /// A specialised row states more than one instantiation and a predicate
    /// over the values a fire binds — see
    /// [`crate::device::Specialisation`]. Every way that can go wrong is
    /// drift between the terms and the row they are written against, which
    /// [`crate::device::Specialisation::agrees`] checks with no device at
    /// all, so reaching this variant means a table changed without the check
    /// being run.
    ///
    /// **It is a refusal and not a fall back to the base row, and that is the
    /// design's central rule applied to its newest feature.** Firing the base
    /// would produce the right number: the two arms are contracted to compute
    /// the same thing, so a broken predicate costs speed and not correctness.
    /// It would also be indistinguishable from a specialisation that never
    /// applies — a fast path silently switched off for the life of the
    /// process, discovered as a performance regression months later. A
    /// failure has to be a refusal rather than a plausible number, and a
    /// plausible number is exactly what falling back would give.
    Specialise {
        /// The row whose specialisation did not work out — the base symbol
        /// when the predicate is at fault, the variant's when the reshape is.
        symbol: &'static str,
        /// What did not line up.
        why: String,
    },
    /// The driver refused.
    ///
    /// `code` is the `CUresult` as an integer and `why` is its own spelling
    /// of itself. Both, because the number is what a bug report can be
    /// grepped for and the name is what makes the report legible — and
    /// because `CUresult` is `cudarc`'s type, which this error must not put
    /// in the signature of a variant that a feature-gated-off consumer may
    /// still want to match on.
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
            Error::Unknown { symbol } => write!(f, "no unit hosts `{symbol}`"),
            Error::NoDevice => {
                write!(f, "no CUDA device is current, so no architecture could be discovered")
            }
            Error::Compile { unit, why } => write!(f, "`{unit}` would not compile: {why}"),
            Error::Missing { unit, symbol } => {
                write!(f, "`{unit}` compiled and its image has no entry for row `{symbol}`")
            }
            Error::Geometry { symbol, why } => {
                write!(f, "`{symbol}` states a launch these dims cannot satisfy: {why:?}")
            }
            Error::Args(why) => write!(f, "{why}"),
            Error::Specialise { symbol, why } => {
                write!(f, "`{symbol}` states a specialisation this fire could not take: {why}")
            }
            Error::Driver { what, code, why } => write!(f, "{what} failed with {code} ({why})"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Args(why) => Some(why),
            _ => None,
        }
    }
}

impl From<crate::runtime::ArgError> for Error {
    fn from(why: crate::runtime::ArgError) -> Self {
        Error::Args(why)
    }
}
