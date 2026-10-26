use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use eta_ir::registry::Stage;
use eta_ir::validate::ValidateError;

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
        Span {
            file: l.file(),
            line: l.line(),
            col: l.column(),
        }
    }
    /// `file:line`.
    pub fn at(&self) -> String {
        alloc::format!("{}:{}", self.file, self.line)
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    Host,
    Stage(Stage),
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

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TraceError {
    DoubleEndpoint {
        channel: Span,
        role: &'static str,
        first: (Endpoint, Span),
        second: (Endpoint, Span),
    },
    ReadinessConflict {
        channel: Span,
        detail: String,
        span: Span,
    },
    SinkMisplacement {
        sink: String,
        stage: Stage,
        span: Span,
    },
    Authoring {
        detail: String,
        span: Span,
    },
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
                "channel {} has two {role} endpoints (SPSC): first {} at {}, second {} at {}",
                channel.at(),
                first.0,
                first.1,
                second.0,
                second.1
            ),
            TraceError::ReadinessConflict {
                channel,
                detail,
                span,
            } => {
                write!(
                    f,
                    "channel {} readiness conflict at {span}: {detail}",
                    channel.at()
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
