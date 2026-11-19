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

    /// The lanes a step refused for their own sake: a lane's word, mask,
    /// positions, adapter or payload this load does not serve as stated.
    /// Nothing of the frame ran; every other lane would have, and the
    /// runtime fails these lanes' fires alone ([`crate::Engine::submit`]).
    #[error("{}", refused_lanes(.0))]
    Lanes(Vec<crate::LaneFault>),
}

fn refused_lanes(lanes: &[crate::LaneFault]) -> String {
    let named: Vec<String> = lanes
        .iter()
        .map(|fault| format!("lane {} refused: {}", fault.lane, fault.reason))
        .collect();
    named.join("; ")
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

    /// Lane `lane` of its step refused, for `why`, with no other lane named.
    pub fn lane(lane: u32, why: impl fmt::Display) -> Error {
        Error::Lanes(vec![crate::LaneFault {
            lane,
            instance: None,
            reason: why.to_string(),
        }])
    }

    /// A refusal of the step's lanes `self` is, read at step `step` of a
    /// frame whose steps before it fired: the frame stops there, answering
    /// those steps ([`crate::FrameTicket::refused`]), and any other
    /// refusal, or one at the first step, refuses the frame.
    #[must_use]
    pub fn stops_frame_at(&self, step: usize) -> bool {
        step > 0 && matches!(self, Error::Lanes(_))
    }

    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Error::Exhausted { .. })
    }
}

pub type Result<T> = std::result::Result<T, Error>;
