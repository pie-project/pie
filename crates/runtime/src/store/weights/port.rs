//! The weight cache's engine side, through the engine's scheduler lane.
//!
//! The governor's tick cannot wait on the lane, so neither verb blocks: a
//! resident set is queued on the lane ahead of every fire queued after it,
//! and the use record a tick reads is the one the lane answered since the
//! last tick asked (one tick late, within a window of
//! [`super::WINDOW_TICKS`]). An engine that answers either verb
//! unsupported is asked nothing more; one with no weight shapes is never
//! asked at all. The load alone waits: on the resident set the engine
//! starts with, before the first fire can be submitted.

use ::engine::chunk::{WeightResidency, WeightUses};
use tokio::sync::oneshot::{Receiver, error::TryRecvError};

use super::WeightPort;
use crate::scheduler::worker::HostKvOp;

pub type Answer = Receiver<anyhow::Result<Vec<u8>>>;

/// See the module docs.
pub struct EnginePort {
    engine: usize,
    /// Nothing streams through this engine: nothing to say, nothing to ask.
    quiet: bool,
    /// The engine does not serve the weight verbs.
    unsupported: bool,
    /// The use record asked for and not yet read.
    uses: Option<Answer>,
    /// Resident sets queued whose answers are not read yet.
    residencies: Vec<Answer>,
}

impl EnginePort {
    /// The port to `engine`, quiet when it has no weight shapes.
    #[must_use]
    pub const fn new(engine: usize, quiet: bool) -> Self {
        Self {
            engine,
            quiet,
            unsupported: false,
            uses: None,
            residencies: Vec::new(),
        }
    }

    /// Whether the engine is asked nothing (no weight shapes, or it does
    /// not serve the verbs).
    #[must_use]
    pub const fn idle(&self) -> bool {
        self.quiet || self.unsupported
    }

    /// Reads a lane's answer: an unsupported engine is asked nothing more,
    /// any other refusal is logged.
    fn settle(&mut self, answer: anyhow::Result<Vec<u8>>) -> Option<Vec<u8>> {
        match answer {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                if let Some(::engine::Error::Unsupported { .. }) = error.downcast_ref() {
                    tracing::debug!(engine = self.engine, %error, "no weight streaming");
                    self.unsupported = true;
                } else {
                    tracing::warn!(engine = self.engine, %error, "a weight verb failed");
                }
                None
            }
        }
    }

    /// The resident sets queued and not answered yet, for a caller that
    /// must know the engine took them; each answer comes back through
    /// [`Self::answered`].
    pub fn take_queued(&mut self) -> Vec<Answer> {
        std::mem::take(&mut self.residencies)
    }

    /// Reads an answer [`Self::take_queued`] handed out.
    ///
    /// # Errors
    ///
    /// The engine refused the resident set. An unsupported answer is no
    /// error: the port is asked nothing more.
    pub fn answered(&mut self, answer: anyhow::Result<Vec<u8>>) -> anyhow::Result<()> {
        match answer {
            Ok(_) => Ok(()),
            Err(error)
                if matches!(
                    error.downcast_ref(),
                    Some(::engine::Error::Unsupported { .. })
                ) =>
            {
                self.settle(Err(error));
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn queue(&self, op: HostKvOp) -> ::engine::Result<Answer> {
        crate::scheduler::scheduler_handle(self.engine)
            .and_then(|scheduler| scheduler.host_kv_queued(op))
            .map_err(::engine::Error::device)
    }
}

impl WeightPort for EnginePort {
    fn uses(&mut self) -> ::engine::Result<WeightUses> {
        if self.idle() {
            return Ok(WeightUses::default());
        }
        let mut read = WeightUses::default();
        if let Some(mut pending) = self.uses.take() {
            match pending.try_recv() {
                Ok(answer) => {
                    if let Some(bytes) = self.settle(answer) {
                        read = serde_json::from_slice(&bytes).unwrap_or_default();
                    }
                }
                Err(TryRecvError::Empty) => {
                    self.uses = Some(pending);
                    return Ok(read);
                }
                Err(TryRecvError::Closed) => {}
            }
        }
        if !self.idle() {
            self.uses = Some(self.queue(HostKvOp::WeightUses)?);
        }
        Ok(read)
    }

    fn set_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()> {
        let mut answered = Vec::new();
        self.residencies
            .retain_mut(|pending| match pending.try_recv() {
                Ok(answer) => {
                    answered.push(answer);
                    false
                }
                Err(TryRecvError::Empty) => true,
                Err(TryRecvError::Closed) => false,
            });
        for answer in answered {
            self.settle(answer);
        }
        if self.idle() {
            return Ok(());
        }
        let pending = self.queue(HostKvOp::WeightResidency(residency.clone()))?;
        self.residencies.push(pending);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quiet_port_asks_nothing_and_reads_no_uses() {
        // Engine 9_999 has no scheduler: a port that asked would fail.
        let mut port = EnginePort::new(9_999, true);
        assert!(port.idle());
        assert_eq!(port.uses().unwrap(), WeightUses::default());
        port.set_residency(&WeightResidency::default()).unwrap();
    }

    #[test]
    fn a_refused_starting_residency_is_an_error_and_unsupported_is_not() {
        let mut port = EnginePort::new(9_999, false);
        let refused = anyhow::anyhow!("the engine is out of staging");
        assert!(port.answered(Err(refused)).is_err());
        assert!(!port.idle());
        let unsupported = anyhow::Error::from(::engine::Error::unsupported("plain", "weight"));
        port.answered(Err(unsupported)).unwrap();
        assert!(port.idle(), "an unsupported engine is asked nothing more");
        port.answered(Ok(Vec::new())).unwrap();
    }

    #[test]
    fn an_unsupported_answer_quiets_the_port() {
        let mut port = EnginePort::new(9_999, false);
        assert!(!port.idle());
        let refused = anyhow::Error::from(::engine::Error::unsupported("plain", "weight_uses"));
        assert!(port.settle(Err(refused)).is_none());
        assert!(port.idle(), "an unsupported engine is asked nothing more");
        assert_eq!(port.uses().unwrap(), WeightUses::default());
        port.set_residency(&WeightResidency::default()).unwrap();
    }
}
