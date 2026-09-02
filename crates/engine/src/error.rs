//! `Error` — what an engine says when it will not do the thing.
//!
//! [`Error::Exhausted`] and [`Error::Impossible`] are scheduling answers, not
//! faults: a fire that does not fit right now is `Exhausted` (retryable); one
//! that never fits is `Impossible`.

use std::fmt;

/// What an engine answers when a verb does not complete. One enum for every
/// verb, so the runtime's dispatch loop wants one `match`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// This engine does not serve this verb at all (e.g. Metal has no
    /// `copy_kv`); the runtime matches on this rather than logging it.
    #[error("the {engine} engine does not serve `{verb}`")]
    Unsupported {
        /// The verb, spelled as the trait method.
        verb: &'static str,
        /// The engine kind that refused it, as [`Engine::kind`](crate::Engine::kind).
        engine: &'static str,
    },

    /// The submission is malformed (e.g. a CSR that decreases, a lane naming
    /// a slot the pools do not have); retrying verbatim cannot help.
    #[error("invalid submission: {0}")]
    Invalid(String),

    /// A handle the submission names is closed, or was never open.
    #[error("{what} {id} is closed")]
    Closed {
        /// What kind of handle — `"instance"`, `"channel"`, `"program"`.
        what: &'static str,
        /// Its id.
        id: u64,
    },

    /// The device cannot fit this fire now, but a later one could.
    /// `wanted`/`available` are in whatever unit `resource` names.
    #[error("{resource} exhausted: wanted {wanted}, {available} available")]
    Exhausted {
        /// Which pool ran out.
        resource: &'static str,
        /// How much the submission asked for.
        wanted: u64,
        /// How much there was.
        available: u64,
    },

    /// The device can never fit this fire: it is past a ceiling this load was
    /// baked against, and no amount of freeing helps.
    #[error("impossible submission: {0}")]
    Impossible(String),

    /// The load failed: a checkpoint that does not fit the plan, a plan these
    /// budgets do not admit, a device that would not bind.
    #[error("load failed: {0}")]
    Load(String),

    /// A guest program would not compile, adopt, or bind.
    #[error("program: {0}")]
    Program(String),

    /// The device itself failed — an allocation, a launch, a transfer, a
    /// synchronize. The message is the backend's, verbatim.
    #[error("device: {0}")]
    Device(String),

    /// The engine is gone: the remote hung up, the process died, the shell was
    /// torn down under an outstanding ticket.
    #[error("engine disconnected: {0}")]
    Disconnected(String),
}

impl Error {
    /// The refusal a default trait-method body answers with.
    #[must_use]
    pub const fn unsupported(engine: &'static str, verb: &'static str) -> Error {
        Error::Unsupported { verb, engine }
    }

    /// Build an [`Error::Invalid`] from anything that renders.
    pub fn invalid(why: impl fmt::Display) -> Error {
        Error::Invalid(why.to_string())
    }

    /// Build an [`Error::Device`] from anything that renders.
    pub fn device(why: impl fmt::Display) -> Error {
        Error::Device(why.to_string())
    }

    /// Build an [`Error::Program`] from anything that renders.
    pub fn program(why: impl fmt::Display) -> Error {
        Error::Program(why.to_string())
    }

    /// Build an [`Error::Load`] from anything that renders.
    pub fn load(why: impl fmt::Display) -> Error {
        Error::Load(why.to_string())
    }

    /// True for the two variants that are scheduling answers rather than
    /// failures — see the module header.
    #[must_use]
    pub const fn is_scheduling(&self) -> bool {
        matches!(
            self,
            Error::Exhausted { .. } | Error::Impossible(_)
        )
    }

    /// True only for [`Error::Exhausted`]: legal to resubmit unchanged once
    /// something frees what it wanted. Never applies past admission.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Error::Exhausted { .. })
    }
}

/// What every [`Engine`](crate::Engine) verb answers.
pub type Result<T> = std::result::Result<T, Error>;
