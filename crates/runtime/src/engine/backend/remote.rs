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

pub struct RemoteEngine {
    peer: String,
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

#[derive(Clone)]
pub struct RemoteDisconnectHandle {
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

impl RemoteDisconnectHandle {
    pub fn disconnect(&self, message: impl Into<String>) {
        if self.connected.swap(false, Ordering::AcqRel) {
            self.broker.close_all(message);
            self.disconnected.notify_waiters();
        }
    }

    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }
}

impl RemoteEngine {
    #[must_use]
    pub fn new(peer: impl Into<String>) -> RemoteEngine {
        RemoteEngine {
            peer: peer.into(),
            broker: CompletionBroker::new(),
            connected: Arc::new(AtomicBool::new(true)),
            disconnected: Arc::new(tokio::sync::Notify::new()),
        }
    }

    #[must_use]
    pub fn peer(&self) -> &str {
        &self.peer
    }

    #[must_use]
    pub fn disconnect_handle(&self) -> RemoteDisconnectHandle {
        RemoteDisconnectHandle {
            broker: self.broker.clone(),
            connected: Arc::clone(&self.connected),
            disconnected: Arc::clone(&self.disconnected),
        }
    }

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

    fn expect_fire(&mut self, submission: &Step) {
        let _ = submission;
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        let _ = registration;
        Err(self.refuse("register_program"))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> Result<RegisteredChannel> {
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
