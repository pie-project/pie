//! The remote seam — a driver that is not here yet, and says so.
//!
//! # `palo B-remote`: the envelope died with `driver_api::remote`
//!
//! What stood here was a 539-line tarpc client: `ExecutorRpcClient`,
//! `ExecutorRequest`/`ExecutorResponse`, `RemoteLaunch`, `RemoteBindInstance`,
//! `RemoteRegisterChannel`, `RemoteChannelValue`, `ScratchGrant`,
//! `TerminalCellState`, `RemoteError`/`RemoteErrorKind`,
//! `REMOTE_WIRE_VERSION` — every one of them a type in `driver-api::remote`,
//! and every one of them deleted by the palo contract rewrite. The rewrite's
//! ruling (design §7, decision 19) is one sentence:
//!
//! > Remote is a property, not an encoding: every noun serde, trait
//! > object-safe; wire versioning is the transport's concern, not the
//! > contract's.
//!
//! So a remote driver is **a type in the transport that implements
//! [`Driver`] and whose method bodies happen to be round trips**. Designing
//! that envelope is not this wave, and neither is guessing at it: a stub that
//! answered `Ok(())` to `fire` would be an engine that drops every offloaded
//! request silently, which is the one failure mode this file exists to
//! prevent.
//!
//! # What the future envelope has to carry
//!
//! Recorded here rather than in a commit message, because the next wave reads
//! this file first. Each verb's marker below names its own half; the shared
//! frame is:
//!
//! * **identity and admission** — which `Plan` the peer loaded, which
//!   [`Capabilities`](driver_api::Capabilities) it answered, and the scratch
//!   grant (base page + count) the caller may address inside its pool. The
//!   old `ScratchGrant` + `HelloRequest`/`HelloResponse` pair.
//! * **a wire version, on the TRANSPORT** — `REMOTE_WIRE_VERSION` was a
//!   `driver-api` constant checked at hello; it belongs where the bytes are.
//! * **liveness** — the disconnect notification that closes every outstanding
//!   completion at once. That half is NOT dead and is kept below, because it
//!   is engine bookkeeping (the broker) rather than an encoding.
//! * **an asynchronous ticket** — [`FireTicket`](driver_api::FireTicket) is
//!   answered with an empty `readouts` by a driver that answers before the
//!   device is done, and `FireTicket::id` is what the engine-side broker
//!   correlates the later completion on. A remote driver is the first one
//!   that needs that path; the shells in this workspace are synchronous.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use driver_api::channel::{ChannelRegistration, RegisteredChannel};
use driver_api::error::{DriverError, Result};
use driver_api::fire::{FireSubmission, FireTicket, MediaEncode};
use driver_api::load::{LoadRequest, Loaded};
use driver_api::program::{BoundInstance, InstanceBinding, InstanceId, ProgramId, ProgramRegistration};
use driver_api::transfer::{KvCopy, PoolResize, StateCopy};
use driver_api::{ChannelId, Driver};

use crate::driver::CompletionBroker;

/// A registered peer whose transport has not been built.
///
/// It is a real registry entry — the engine's KV store, scheduler lane and
/// partner tables all key on a `DriverId`, and a peer that could not be
/// registered at all would be a peer the offload planner cannot even see to
/// refuse. Every verb answers [`DriverError::Unsupported`].
pub struct RemoteDriver {
    /// Who this was meant to reach, for the refusal's message.
    peer: String,
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

/// The half of the old remote driver that survives: closing every outstanding
/// completion when the peer goes away.
///
/// This is engine bookkeeping, not an encoding — it is the broker's
/// `close_all` behind a handle the link layer can hold — so it is kept
/// whole.
#[derive(Clone)]
pub struct RemoteDisconnectHandle {
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    disconnected: Arc<tokio::sync::Notify>,
}

impl RemoteDisconnectHandle {
    /// Fail every completion this peer's driver still owes.
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

impl RemoteDriver {
    /// A driver for `peer`, with no transport behind it.
    ///
    /// `new` used to take `(client, runtime, capabilities, grant)` — the four
    /// pieces of the dead envelope. It takes the peer's name, because that is
    /// the only one of the four that survives into every possible successor.
    #[must_use]
    pub fn new(peer: impl Into<String>) -> RemoteDriver {
        RemoteDriver {
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
    fn refuse(&self, verb: &'static str) -> DriverError {
        tracing::warn!(
            peer = %self.peer,
            verb,
            "the remote driver has no transport: palo B-remote redesigns the \
             envelope `driver_api::remote` carried"
        );
        DriverError::unsupported("remote", verb)
    }
}

impl Driver for RemoteDriver {
    fn kind(&self) -> &'static str {
        "remote"
    }

    // palo B-remote: `device_facts` was answered out of the peer's
    // `HelloResponse`. The envelope must carry a `DeviceFacts` — backend name,
    // memory domain WITH the peer's ordinal, sm count, alignment — because the
    // engine stamps a `KvCopy`'s domains from it and a wrong domain is a copy
    // between two unrelated pools.

    fn load(&mut self, request: LoadRequest) -> Result<Loaded> {
        // palo B-remote: the plan crosses here, and it is the one noun the
        // rewrite made cheap — `model_ir::Plan` is serde. What the envelope
        // adds is the CHECKPOINT question: a `Checkpoint::Path` is a path in
        // the PEER's filesystem, and a caller that means its own has to say
        // so.
        let _ = request;
        Err(self.refuse("load"))
    }

    fn fire(&mut self, submission: &FireSubmission) -> Result<FireTicket> {
        // palo B-remote: the whole submission is serde, so the envelope is a
        // framed `FireSubmission` and a `FireTicket` back. The two things it
        // must add: (a) the ticket may arrive EMPTY and be completed later,
        // which is the asynchronous path no shell here exercises; (b) the
        // scheduling refusals — `Exhausted`/`Impossible` — have to survive the
        // round trip as themselves, because the lane loop retries one and
        // drops the other.
        let _ = submission;
        Err(self.refuse("fire"))
    }

    fn register_program(&mut self, registration: &ProgramRegistration) -> Result<ProgramId> {
        // palo B-remote: ids are MINTED BY THE PEER, so the envelope needs the
        // local↔remote id maps the old `RemoteDriver` kept in three
        // `HashMap<u64, u64>`s.
        let _ = registration;
        Err(self.refuse("register_program"))
    }

    fn register_channel(&mut self, registration: &ChannelRegistration) -> Result<RegisteredChannel> {
        // palo B-remote: the ring is the PEER's and the wait slots are the
        // caller's, which is what the old `RemoteChannelBinding` was trying to
        // say by shipping addresses. See `crate::driver::channel` on where the
        // host ring lives now.
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
        // palo B-remote: this is the verb the offload plane is actually about
        // — pushing pages into a peer's scratch grant. The envelope must carry
        // either the bytes (the old `InlineKvPayload`/`PushKv`) or an RDMA
        // handle (`KvHandle` + the peer's registration), and which of the two
        // is a deployment's choice, not the contract's.
        let _ = copy;
        Err(self.refuse("copy_kv"))
    }

    fn copy_state(&mut self, copy: &StateCopy) -> Result<()> {
        let _ = copy;
        Err(self.refuse("copy_state"))
    }

    fn resize_pool(&mut self, resize: &PoolResize) -> Result<()> {
        let _ = resize;
        Err(self.refuse("resize_pool"))
    }

    fn encode(&mut self, plan: &mut MediaEncode) -> Result<()> {
        // palo B-remote: the old envelope had a whole media plane here
        // (`RemoteMediaBlob`, `RemoteMediaKind`, a blob-fetch budget per
        // client). `MediaEncode` is serde and carries its own bytes, so the
        // envelope's remaining question is the SIZE ceiling — an encode is
        // megabytes and a frame limit is the transport's.
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
