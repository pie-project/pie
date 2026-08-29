//! Trace-time errors with source spans.
//!
//! Two layers: the SDK **span lints** (double-endpoint, readiness-direction,
//! sink misplacement) caught during assembly with `#[track_caller]` source
//! spans, and the authoritative **bind** verdict wrapping the IR's
//! [`ValidateError`] on the
//! canonical container.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use tensor_ir::registry::Stage;
use tensor_ir::validate::ValidateError;

/// A source location captured at an author call site (`file:line:col`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Path of the source file containing the call site.
    pub file: &'static str,
    /// 1-based line number of the call site.
    pub line: u32,
    /// 1-based column number of the call site.
    pub col: u32,
}

impl Span {
    /// Captures the current caller's source location as a [`Span`].
    #[track_caller]
    pub fn here() -> Span {
        let l = core::panic::Location::caller();
        Span {
            file: l.file(),
            line: l.line(),
            col: l.column(),
        }
    }
    /// Builds a [`Span`] from an already-captured caller
    /// [`Location`](core::panic::Location).
    pub fn of(l: &'static core::panic::Location<'static>) -> Span {
        Span {
            file: l.file(),
            line: l.line(),
            col: l.column(),
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.col)
    }
}

/// The endpoint that touched a channel (SPSC has one producer + one consumer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The host engine side (async `put`/`take` across the engine boundary).
    Host,
    /// A traced device stage, named by its [`Stage`].
    Stage(Stage),
    /// The forward's descriptor ports (e.g. `embed` consuming tokens).
    Descriptor,
}

impl fmt::Display for Endpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Endpoint::Host => write!(f, "host"),
            Endpoint::Stage(k) => write!(f, "stage `{}`", k.name()),
            Endpoint::Descriptor => write!(f, "descriptor port"),
        }
    }
}

/// A trace-time error. SDK span lints carry the source [`Span`]; the bind
/// verdict carries the IR's authoritative [`ValidateError`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TraceError {
    /// A channel got a second distinct producer or consumer endpoint (SPSC).
    DoubleEndpoint {
        /// Name of the channel that acquired a second endpoint.
        channel: String,
        /// Label for the endpoint role claimed twice, as printed (e.g. `"host"`).
        role: &'static str,
        /// The first claim of `role`: which [`Endpoint`], and where.
        first: (Endpoint, Span),
        /// The conflicting second claim of `role`: which [`Endpoint`], and where.
        second: (Endpoint, Span),
    },
    /// A channel is consumed but never produced or seeded (its take can never
    /// become full — a readiness-direction conflict).
    ReadinessConflict {
        /// Name of the channel consumed with no producer or seed.
        channel: String,
        /// Why readiness can never be reached, phrased for the author.
        detail: String,
        /// Call site of the consume that can never complete.
        span: Span,
    },
    /// A configuration sink at a stage that does not precede its consumption
    /// point.
    SinkMisplacement {
        /// Name of the misplaced configuration sink.
        sink: String,
        /// The [`Stage`] the sink was attached to (too late to precede its
        /// consumer).
        stage: Stage,
        /// Call site of the sink placement.
        span: Span,
    },
    /// The trace asked for something the op vocabulary cannot express, or
    /// combined operands whose shapes or dtypes do not agree.
    ///
    /// Reported rather than panicked. `tensor-dsl` traces inside a
    /// `wasm32-wasip2` guest, where a panic is a trap: the inferlet aborts
    /// and the author gets a stack-less abort code instead of the file and
    /// line of the call that was wrong. Recording lets the recorder keep
    /// going and [`Builder::build`](crate::builder::Builder::build) return
    /// every authoring mistake in the pass at once, each with its span.
    Authoring {
        /// What was wrong, phrased for the author of the trace.
        detail: String,
        /// The call site that recorded it.
        span: Span,
    },
    /// The authoritative bind verdict (the IR's validator on the canonical bytes).
    Bind(ValidateError),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::DoubleEndpoint {
                channel,
                role,
                first,
                second,
            } => write!(
                f,
                "channel `{channel}` has two {role} endpoints (SPSC): first {} at {}, second {} at {}",
                first.0, first.1, second.0, second.1
            ),
            TraceError::ReadinessConflict {
                channel,
                detail,
                span,
            } => {
                write!(
                    f,
                    "channel `{channel}` readiness conflict at {span}: {detail}"
                )
            }
            TraceError::SinkMisplacement { sink, stage, span } => write!(
                f,
                "sink `{sink}` misplaced in stage `{}` at {span}: it does not precede the point consuming its effect",
                stage.name()
            ),
            TraceError::Authoring { detail, span } => {
                write!(f, "{span}: {detail}")
            }
            TraceError::Bind(e) => write!(f, "bind failed: {e}"),
        }
    }
}

/// A bundle of trace-time errors (assembly collects all it can before failing).
#[derive(Clone, Debug, PartialEq)]
pub struct TraceErrors(pub Vec<TraceError>);

impl fmt::Display for TraceErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "trace failed with {} error(s):", self.0.len())?;
        for e in &self.0 {
            writeln!(f, "  - {e}")?;
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TraceErrors {}
