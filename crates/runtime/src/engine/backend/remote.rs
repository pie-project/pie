//! The remote seam — an engine that is not here yet, and says so. A stub
//! that answered `Ok(())` would silently drop every offloaded request, which
//! is the failure mode this file exists to prevent: every verb refuses by
//! name. The disconnect/liveness half is real runtime bookkeeping (the
//! broker) and is kept live rather than stubbed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::channel::{ChannelRegistration, RegisteredChannel};
use engine::error::{Error, Result};
use engine::fire::{FrameSubmission, FrameTicket, MediaEncode, Step};
use engine::load::{LoadRequest, Loaded};
use engine::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use engine::transfer::{KvCopy, StateCopy};
use engine::{ChannelId, Engine};

use crate::engine::CompletionBroker;

/// A registered peer whose transport has not been built — a real registry
/// entry so the offload planner can see it to refuse. Every verb answers
/// [`Error::Unsupported`].
pub struct RemoteEngine {
    /// Who this was meant to reach, for the refusal's message.
    peer: String,
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

/// Closes every outstanding completion when the peer goes away — the
/// broker's `close_all` behind a handle the link layer can hold.
#[derive(Clone)]
pub struct RemoteDisconnectHandle {
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

impl RemoteDisconnectHandle {
    /// Fail every completion this peer's engine still owes.
    pub fn disconnect(&self, message: impl Into<String>) {
        if self.connected.swap(false, Ordering::AcqRel) {
            self.broker.close_all(message);
            self.disconnected.notify_waiters();
        }
    }

    /// Whether the peer is still believed to be there.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }
}

impl RemoteEngine {
    /// An engine for `peer`, with no transport behind it.
    #[must_use]
    pub fn new(peer: impl Into<String>) -> RemoteEngine {
        RemoteEngine {
            peer: peer.into(),
            broker: CompletionBroker::new(),
            connected: Arc::new(AtomicBool::new(true)),
            disconnected: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Which peer this addresses.
    #[must_use]
    pub fn peer(&self) -> &str {
        &self.peer
    }

    /// The handle that fails this peer's outstanding work.
    #[must_use]
    pub fn disconnect_handle(&self) -> RemoteDisconnectHandle {
        RemoteDisconnectHandle {
            broker: self.broker.clone(),
            connected: Arc::clone(&self.connected),
            disconnected: Arc::clone(&self.disconnected),
        }
    }

    /// The refusal every verb answers, with the peer named.
    fn refuse(&self, verb: &'static str) -> Error {
        tracing::warn!(
            peer = %self.peer,
            verb,
            "the remote engine has no transport: remote executors are not \
             supported in this release"
        );
        Error::unsupported("remote", verb)
    }
}

impl Engine for RemoteEngine {
    fn kind(&self) -> &'static str {
        "remote"
    }

    fn load(&mut self, request: LoadRequest) -> Result<Loaded> {
        let _ = request;
        Err(self.refuse("load"))
    }

    fn submit(&mut self, frame: &FrameSubmission) -> Result<FrameTicket> {
        let _ = frame;
        Err(self.refuse("submit"))
    }

    // An advisory does not earn a round trip: shipping the hint over a
    // transport would cost more than the host work it saves, so this is an
    // explicit no-op rather than a refusal.
    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.refuse("register_program"))
    }

    fn register_channel(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel> {
        let _ = registration;
        Err(self.refuse("register_channel"))
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> Result<BoundInstance> {
        let _ = binding;
        Err(self.refuse("bind_instance"))
    }

    fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let _ = id;
        Err(self.refuse("close_instance"))
    }

    fn close_channel(&mut self, id: ChannelId) -> Result<()> {
        let _ = id;
        Err(self.refuse("close_channel"))
    }

    fn copy_kv(&mut self, copy: &KvCopy) -> Result<()> {
        let _ = copy;
        Err(self.refuse("copy_kv"))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.refuse("copy_state"))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        let _ = plan;
        Err(self.refuse("encode"))
    }

    fn disconnect(&self, message: &str) {
        if self.connected.swap(false, Ordering::AcqRel) {
            self.broker.close_all(message.to_string());
            self.disconnected.notify_waiters();
        }
    }
}
