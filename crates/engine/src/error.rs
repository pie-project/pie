use std::fmt;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("the {engine} engine does not serve `{verb}`")]
    Unsupported {
        verb: &'static str,
        engine: &'static str,
    },

    #[error("invalid submission: {0}")]
    Invalid(String),

    #[error("{what} {id} is closed")]
    Closed { what: &'static str, id: u64 },

    #[error("{resource} exhausted: wanted {wanted}, {available} available")]
    Exhausted {
        resource: &'static str,
        wanted: u64,
        available: u64,
    },

    #[error("impossible submission: {0}")]
    Impossible(String),

    #[error("load failed: {0}")]
    Load(String),

    #[error("program: {0}")]
    Program(String),

    #[error("device: {0}")]
    Device(String),

    #[error("engine disconnected: {0}")]
    Disconnected(String),
}

impl Error {
    #[must_use]
    pub const fn unsupported(engine: &'static str, verb: &'static str) -> Error {
        Error::Unsupported { verb, engine }
    }

    pub fn invalid(why: impl fmt::Display) -> Error {
        Error::Invalid(why.to_string())
    }

    pub fn device(why: impl fmt::Display) -> Error {
        Error::Device(why.to_string())
    }

    pub fn program(why: impl fmt::Display) -> Error {
        Error::Program(why.to_string())
    }

    pub fn load(why: impl fmt::Display) -> Error {
        Error::Load(why.to_string())
    }

    #[must_use]
    pub const fn is_scheduling(&self) -> bool {
        matches!(self, Error::Exhausted { .. } | Error::Impossible(_))
    }

    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Error::Exhausted { .. })
    }
}

pub type Result<T> = std::result::Result<T, Error>;
