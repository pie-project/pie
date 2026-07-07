//! Trace-time errors with source spans.
//!
//! Two layers: the SDK **span lints** (double-endpoint, readiness-direction,
//! sink misplacement) caught during assembly with `#[track_caller]` source
//! spans (overview P1.3), and the authoritative **bind** verdict wrapping echo's
//! [`ValidateError`](pie_sampling_ir::ptir::validate::ValidateError) on the
//! canonical container.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use pie_sampling_ir::ptir::registry::Stage;
use pie_sampling_ir::ptir::validate::ValidateError;

/// A source location captured at an author call site (`file:line:col`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub file: &'static str,
    pub line: u32,
    pub col: u32,
}

impl Span {
    #[track_caller]
    pub fn here() -> Span {
        let l = core::panic::Location::caller();
        Span { file: l.file(), line: l.line(), col: l.column() }
    }
    pub fn of(l: &'static core::panic::Location<'static>) -> Span {
        Span { file: l.file(), line: l.line(), col: l.column() }
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
    Host,
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

/// A trace-time error. SDK span lints carry the source [`Span`](s); the bind
/// verdict carries echo's authoritative [`ValidateError`].
#[derive(Clone, Debug, PartialEq)]
pub enum TraceError {
    /// A channel got a second distinct producer or consumer endpoint (SPSC, T2).
    DoubleEndpoint {
        channel: String,
        role: &'static str,
        first: (Endpoint, Span),
        second: (Endpoint, Span),
    },
    /// A channel is consumed but never produced or seeded (its take can never
    /// become full — a readiness-direction conflict, T3).
    ReadinessConflict { channel: String, detail: String, span: Span },
    /// A configuration sink at a stage that does not precede its consumption
    /// point (T11).
    SinkMisplacement { sink: String, stage: Stage, span: Span },
    /// The authoritative bind verdict (echo's validator on the canonical bytes).
    Bind(ValidateError),
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::DoubleEndpoint { channel, role, first, second } => write!(
                f,
                "channel `{channel}` has two {role} endpoints (SPSC): first {} at {}, second {} at {}",
                first.0, first.1, second.0, second.1
            ),
            TraceError::ReadinessConflict { channel, detail, span } => {
                write!(f, "channel `{channel}` readiness conflict at {span}: {detail}")
            }
            TraceError::SinkMisplacement { sink, stage, span } => write!(
                f,
                "sink `{sink}` misplaced in stage `{}` at {span}: it does not precede the point consuming its effect",
                stage.name()
            ),
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
