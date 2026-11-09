mod data_transfer;
mod handler;
pub(crate) mod inbox;

pub use data_transfer::InFlightUpload;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use client_api::message::{ClientMessage, ServerMessage as WireServerMessage};
use dashmap::DashMap;
use tokio::sync::{Mutex as TokioMutex, mpsc};

use crate::inferlet::process;
use crate::inferlet::{ProcessEvent, ProcessId};
use crate::service::{ServiceHandler, ServiceMap};

pub type ClientId = u32;

static STATE: OnceLock<Arc<ServerState>> = OnceLock::new();
static SESSION_OUTBOX: LazyLock<
    DashMap<ClientId, Arc<TokioMutex<mpsc::Receiver<WireServerMessage>>>>,
> = LazyLock::new(DashMap::new);

fn install_state(max_upload_bytes: usize, allow_upload: bool) -> Arc<ServerState> {
    if let Some(state) = STATE.get() {
        return Arc::clone(state);
    }

    let state = Arc::new(ServerState {
        next_client_id: AtomicU32::new(1),
        max_upload_bytes,
        allow_upload,
    });
    let _ = STATE.set(Arc::clone(&state));
    STATE.get().cloned().unwrap_or(state)
}

fn get_state() -> Result<Arc<ServerState>> {
    STATE
        .get()
        .cloned()
        .ok_or_else(|| anyhow!("server not initialized; call server::init first"))
}

/// `allow_upload` false refuses every client's program: the system's alone
/// run.
pub(crate) fn init(max_upload_bytes: usize, allow_upload: bool) {
    let _ = install_state(max_upload_bytes, allow_upload);
    inbox::spawn();
}

pub const DEFAULT_TENANT: &str = "default";
pub const DEFAULT_USER: &str = "internal";

pub fn open_session() -> Result<ClientId> {
    open_session_for(DEFAULT_TENANT.to_string(), DEFAULT_USER.to_string())
}

pub fn open_session_for(tenant: String, username: String) -> Result<ClientId> {
    let state = get_state()?;
    let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

    let (out_tx, out_rx) = mpsc::channel(1000);
    SESSION_OUTBOX.insert(id, Arc::new(TokioMutex::new(out_rx)));

    let session = Session::new_inproc(id, state, out_tx, tenant, username);
    CLIENT_SERVICES.spawn(id, || session)?;
    Ok(id)
}

pub fn close_session(client_id: ClientId) {
    SESSION_OUTBOX.remove(&client_id);
    CLIENT_SERVICES.remove(&client_id);
    tracing::debug!(client_id, "session closed");
}

pub fn send_client_message(client_id: ClientId, msg: ClientMessage) -> Result<()> {
    CLIENT_SERVICES.send(&client_id, SessionMessage::ClientRequest(msg))
}

pub async fn recv_messages(
    client_id: ClientId,
    max_wait_ms: u64,
    max_messages: usize,
) -> Result<Vec<WireServerMessage>> {
    let outbox = SESSION_OUTBOX
        .get(&client_id)
        .map(|entry| Arc::clone(entry.value()))
        .ok_or_else(|| anyhow!("unknown session {client_id}"))?;

    let mut receiver = outbox.lock().await;
    let mut out = Vec::new();

    if max_messages == 0 {
        return Ok(out);
    }

    if max_wait_ms == 0 {
        while out.len() < max_messages {
            match receiver.try_recv() {
                Ok(msg) => out.push(msg),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        return Ok(out);
    }

    match crate::rt::time::timeout(Duration::from_millis(max_wait_ms), receiver.recv()).await {
        Ok(Some(first)) => out.push(first),
        Ok(None) => return Ok(out),
        Err(_) => return Ok(out),
    }

    while out.len() < max_messages {
        match receiver.try_recv() {
            Ok(msg) => out.push(msg),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
        }
    }

    Ok(out)
}

static CLIENT_SERVICES: LazyLock<ServiceMap<ClientId, SessionMessage>> =
    LazyLock::new(ServiceMap::new);

/// Where a process's end left its handles. A session gone by now drops them.
pub(crate) fn handles_settled(
    client_id: ClientId,
    settled: Vec<(client_api::HandleName, crate::handle::Landing)>,
) {
    let _ = CLIENT_SERVICES.send(&client_id, SessionMessage::HandlesSettled(settled));
}

/// The worker policy dropped the handle it knows as `id`.
pub(crate) fn discard_handle(client_id: ClientId, id: ProcessId, why: String) {
    let _ = CLIENT_SERVICES.send(&client_id, SessionMessage::DiscardHandle(id, why));
}

/// The turn behind `corr_id` is cancelled before it had a process.
pub fn forget_frame(client_id: ClientId, corr_id: u32) {
    let _ = CLIENT_SERVICES.send(&client_id, SessionMessage::Forget(corr_id));
}

pub(crate) fn send_event(
    client_id: ClientId,
    process_id: ProcessId,
    event: &ProcessEvent,
) -> Result<()> {
    CLIENT_SERVICES.send(
        &client_id,
        SessionMessage::Event {
            process_id,
            event: event.name().to_string(),
            value: event.value().to_string(),
        },
    )
}

pub(crate) fn send_file(
    client_id: ClientId,
    process_id: ProcessId,
    data: Bytes,
    name: Option<String>,
) -> Result<()> {
    CLIENT_SERVICES.send(
        &client_id,
        SessionMessage::File {
            process_id,
            data,
            name,
        },
    )
}

pub(crate) async fn receive_file(client_id: ClientId, process_id: ProcessId) -> Result<Bytes> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    CLIENT_SERVICES.send(
        &client_id,
        SessionMessage::ReceiveFile {
            process_id,
            sender: tx,
        },
    )?;
    Ok(rx.await?)
}

struct ServerState {
    next_client_id: AtomicU32,
    pub(super) max_upload_bytes: usize,
    pub(super) allow_upload: bool,
}

#[derive(Debug)]
enum SessionMessage {
    Event {
        process_id: ProcessId,
        event: String,
        value: String,
    },
    File {
        process_id: ProcessId,
        data: Bytes,
        name: Option<String>,
    },
    ClientRequest(ClientMessage),
    HandlesSettled(Vec<(client_api::HandleName, crate::handle::Landing)>),
    DiscardHandle(ProcessId, String),
    /// A turn was cancelled: if its frame is still waiting, it never runs.
    Forget(u32),
    ReceiveFile {
        process_id: ProcessId,
        sender: tokio::sync::oneshot::Sender<Bytes>,
    },
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(super) enum UploadKey {
    Program(u32),
    File(ProcessId, String),
}

pub(super) const MAX_INFLIGHT_UPLOADS: usize = 16;

struct Session {
    pub(super) id: ClientId,
    pub(super) username: String,
    pub(super) tenant: String,
    state: Arc<ServerState>,
    pub(super) inflight_uploads: DashMap<UploadKey, InFlightUpload>,
    pub(super) attached_processes: Vec<ProcessId>,
    /// The subset of `attached_processes` this session launched with output
    /// capture — the launcher owns them, so its close ends them (below);
    /// processes merely attached to are another session's and only detach.
    pub(super) launched_processes: HashSet<ProcessId>,
    pub(super) file_waiters: HashMap<ProcessId, tokio::sync::oneshot::Sender<Bytes>>,
    pub(super) handles: crate::handle::Table,
    /// Frames naming a lent handle, or a handle an earlier frame here names,
    /// in order: each handle's frames run as sent, once it lands.
    pub(super) waiting: VecDeque<ClientMessage>,
    out_tx: mpsc::Sender<WireServerMessage>,
}

impl Session {
    fn new_inproc(
        id: ClientId,
        state: Arc<ServerState>,
        out_tx: mpsc::Sender<WireServerMessage>,
        tenant: String,
        username: String,
    ) -> Self {
        let handles = crate::handle::Table::new(id, tenant.clone());
        Session {
            id,
            username,
            tenant,
            state,
            inflight_uploads: DashMap::new(),
            attached_processes: Vec::new(),
            launched_processes: HashSet::new(),
            file_waiters: HashMap::new(),
            handles,
            waiting: VecDeque::new(),
            out_tx,
        }
    }

    // A process this session launched with capture streams to this client
    // and nobody else, so once the client is gone its output has no reader
    // and its KV seat would otherwise outlive the client for as long as it
    // runs; end it. A process only attached to belongs to another launcher,
    // so it just detaches, as before. A launch with outputs off never joins
    // either list and runs on as a background process.
    fn cleanup(&mut self) {
        for process_id in self.attached_processes.drain(..) {
            if self.launched_processes.contains(&process_id) {
                process::terminate(process_id, Err("client disconnected".to_string()));
            } else {
                process::detach(process_id);
            }
        }

        self.handles.closed();
        SESSION_OUTBOX.remove(&self.id);
        CLIENT_SERVICES.remove(&self.id);
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl ServiceHandler for Session {
    type Message = SessionMessage;

    async fn handle(&mut self, msg: SessionMessage) {
        match msg {
            SessionMessage::ClientRequest(client_msg) => {
                if client_msg.uses_handles() && self.must_wait(&client_msg) {
                    if self.waiting.len() < client_api::MAX_WAITING_HANDLE_FRAMES {
                        self.waiting.push_back(client_msg);
                    } else if let Some(corr_id) = client_msg.corr_id() {
                        let why = "quota: too many frames wait on lent handles".to_string();
                        self.send_response(corr_id, false, why).await;
                    }
                } else {
                    self.handle_client_message(client_msg).await;
                }
            }
            SessionMessage::HandlesSettled(settled) => {
                self.handles.settle(settled);
                self.run_waiting().await;
            }
            SessionMessage::DiscardHandle(id, why) => self.handles.discard(id, why),
            SessionMessage::Forget(corr_id) => {
                self.waiting
                    .retain(|frame| frame.corr_id() != Some(corr_id));
                self.run_waiting().await;
            }
            SessionMessage::Event {
                process_id,
                event,
                value,
            } => {
                self.send_process_event(process_id, &event, value).await;
            }
            SessionMessage::File {
                process_id,
                data,
                name,
            } => {
                self.send_file_download(process_id, data, name).await;
            }
            SessionMessage::ReceiveFile { process_id, sender } => {
                self.file_waiters.insert(process_id, sender);
            }
        }
    }
}

impl Session {
    async fn run_waiting(&mut self) {
        while let Some(at) = self.next_ready() {
            let frame = self.waiting.remove(at).expect("just found");
            self.handle_client_message(frame).await;
        }
    }

    /// The first waiting frame nothing holds up: what it names is settled,
    /// and no frame before it names any of that.
    fn next_ready(&self) -> Option<usize> {
        let mut ahead = HashSet::new();
        for (at, frame) in self.waiting.iter().enumerate() {
            let names = frame.handle_names();
            if !names.iter().any(|name| self.blocked(&ahead, *name)) {
                return Some(at);
            }
            ahead.extend(names);
        }
        None
    }

    fn must_wait(&self, frame: &ClientMessage) -> bool {
        let ahead: HashSet<_> = self
            .waiting
            .iter()
            .flat_map(ClientMessage::handle_names)
            .collect();
        frame
            .handle_names()
            .iter()
            .any(|name| self.blocked(&ahead, *name))
    }

    fn blocked(
        &self,
        ahead: &HashSet<client_api::HandleName>,
        name: client_api::HandleName,
    ) -> bool {
        ahead.contains(&name) || self.handles.lent(name)
    }

    async fn send(&self, msg: WireServerMessage) {
        if self.out_tx.send(msg).await.is_err() {
            tracing::error!("inproc session sink closed for client {}", self.id);
        }
    }

    pub(super) async fn send_response(&self, corr_id: u32, ok: bool, result: String) {
        self.send(WireServerMessage::Response {
            corr_id,
            ok,
            result,
        })
        .await;
    }

    pub(super) async fn send_process_event(
        &self,
        process_id: ProcessId,
        event: &str,
        value: String,
    ) {
        self.send(WireServerMessage::ProcessEvent {
            process_id: process_id.to_string(),
            event: event.to_string(),
            value,
        })
        .await;
    }
}

impl Session {
    async fn handle_client_message(&mut self, message: ClientMessage) {
        match message {
            ClientMessage::AuthIdentify { corr_id, .. } => {
                self.send_response(corr_id, true, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::AuthProve { corr_id, .. } => {
                self.send_response(corr_id, false, "Already authenticated".to_string())
                    .await;
            }

            ClientMessage::CheckProgram { corr_id, inferlet } => {
                self.handle_check_program(corr_id, &inferlet).await
            }

            ClientMessage::Query {
                corr_id,
                subject,
                record,
            } => self.handle_query(corr_id, subject, record).await,

            ClientMessage::AddProgram {
                corr_id,
                program_hash,
                file,
                chunk_index,
                total_chunks,
                chunk_data,
            } => {
                self.handle_add_program(
                    corr_id,
                    program_hash,
                    file,
                    chunk_index,
                    total_chunks,
                    chunk_data,
                )
                .await
            }
            ClientMessage::LaunchProcess {
                corr_id,
                inferlet,
                input,
                capture_outputs,
                props,
                session,
                files,
                handles,
                returns,
                cluster_props,
                executors,
            } => {
                self.handle_launch_process(
                    corr_id,
                    inferlet,
                    input,
                    capture_outputs,
                    props,
                    session,
                    files,
                    handles,
                    returns,
                    cluster_props,
                    executors,
                )
                .await
            }

            ClientMessage::ReadHandle {
                corr_id,
                name,
                format,
                reader,
            } => {
                self.handle_read_handle(corr_id, name, &format, &reader)
                    .await
            }

            ClientMessage::Handles { corr_id, ops } => {
                let answers = self.handles.apply(ops).await;
                self.send_response(corr_id, true, serde_json::Value::from(answers).to_string())
                    .await;
            }

            ClientMessage::AttachProcess {
                corr_id,
                process_id,
            } => {
                self.handle_attach_process(corr_id, process_id).await;
            }

            ClientMessage::TerminateProcess {
                corr_id,
                process_id,
            } => self.handle_terminate_process(corr_id, process_id).await,

            ClientMessage::ListProcesses { corr_id } => {
                self.handle_list_processes(corr_id).await;
            }

            ClientMessage::SignalProcess {
                process_id,
                message,
            } => self.handle_signal_process(process_id, message).await,

            ClientMessage::TransferFile {
                process_id,
                file_hash,
                chunk_index,
                total_chunks,
                chunk_data,
            } => {
                self.handle_transfer_file(
                    process_id,
                    file_hash,
                    chunk_index,
                    total_chunks,
                    chunk_data,
                )
                .await;
            }

            ClientMessage::Ping { corr_id } => {
                self.send_response(corr_id, true, "Pong".to_string()).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inferlet::process::probe::{self, Seen};

    fn launch_on(corr_id: u32, name: client_api::HandleName) -> ClientMessage {
        ClientMessage::LaunchProcess {
            corr_id,
            inferlet: "chat".into(),
            input: String::new(),
            capture_outputs: true,
            props: Default::default(),
            session: None,
            files: Default::default(),
            handles: client_api::Handles::from([("ws".to_string(), name)]),
            returns: Default::default(),
            cluster_props: Default::default(),
            executors: Vec::new(),
        }
    }

    /// A session with handle 1 lent to a process.
    async fn lent_session() -> (Session, mpsc::Receiver<WireServerMessage>) {
        let (tx, rx) = mpsc::channel(8);
        let mut session = Session::new_inproc(
            9,
            install_state(0, true),
            tx,
            "acme".to_string(),
            "al".to_string(),
        );
        session
            .handles
            .apply(vec![client_api::HandleOp::New {
                name: 1,
                kind: client_api::HandleKind::WorkingSet,
            }])
            .await;
        let lease = session
            .handles
            .lend(
                9,
                client_api::Handles::from([("ws".to_string(), 1)]),
                client_api::Handles::new(),
                None,
            )
            .unwrap();
        std::mem::forget(lease);
        (session, rx)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_server_that_takes_no_uploads_refuses_the_first_chunk() {
        let (tx, mut rx) = mpsc::channel(8);
        let state = Arc::new(ServerState {
            next_client_id: AtomicU32::new(1),
            max_upload_bytes: 1024,
            allow_upload: false,
        });
        let mut session = Session::new_inproc(9, state, tx, "acme".into(), "al".into());
        let chunk = |chunk_index| ClientMessage::AddProgram {
            corr_id: 4,
            program_hash: blake3::hash(b"xy").to_hex().to_string(),
            file: "probe.py".to_string(),
            chunk_index,
            total_chunks: 2,
            chunk_data: vec![b'x' + chunk_index as u8],
        };
        session
            .handle(SessionMessage::ClientRequest(chunk(0)))
            .await;
        let Some(WireServerMessage::Response {
            corr_id,
            ok,
            result,
        }) = rx.recv().await
        else {
            panic!("an answer to the upload");
        };
        assert_eq!((corr_id, ok), (4, false));
        assert!(result.contains("no uploads"), "{result}");
        assert!(session.inflight_uploads.is_empty(), "nothing of it is kept");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_frame_cancelled_while_it_waits_never_runs() {
        let (mut session, _rx) = lent_session().await;
        session
            .handle(SessionMessage::ClientRequest(launch_on(5, 1)))
            .await;
        session
            .handle(SessionMessage::ClientRequest(launch_on(6, 1)))
            .await;
        assert_eq!(session.waiting.len(), 2, "both name what is lent");

        session.handle(SessionMessage::Forget(5)).await;
        let left: Vec<_> = session.waiting.iter().map(ClientMessage::corr_id).collect();
        assert_eq!(left, [Some(6)], "the other keeps its place");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_frame_passes_those_waiting_on_other_handles() {
        let (mut session, _rx) = lent_session().await;
        let stat = |corr_id, name| ClientMessage::Handles {
            corr_id,
            ops: vec![client_api::HandleOp::Stat { name }],
        };
        let fork = ClientMessage::Handles {
            corr_id: 7,
            ops: vec![client_api::HandleOp::Fork { name: 2, from: 1 }],
        };
        session.handle(SessionMessage::ClientRequest(fork)).await;
        assert_eq!(session.waiting.len(), 1, "it reads what is lent");

        session
            .handle(SessionMessage::ClientRequest(stat(8, 2)))
            .await;
        assert_eq!(session.waiting.len(), 2, "it names what the fork makes");

        session
            .handle(SessionMessage::ClientRequest(stat(9, 3)))
            .await;
        assert_eq!(
            session.waiting.len(),
            2,
            "an unrelated handle does not wait"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn frames_waiting_on_a_handle_are_bounded() {
        let (mut session, mut rx) = lent_session().await;
        for corr_id in 0..client_api::MAX_WAITING_HANDLE_FRAMES as u32 {
            session
                .handle(SessionMessage::ClientRequest(launch_on(corr_id, 1)))
                .await;
        }
        session
            .handle(SessionMessage::ClientRequest(launch_on(9999, 1)))
            .await;
        assert_eq!(session.waiting.len(), client_api::MAX_WAITING_HANDLE_FRAMES);
        let WireServerMessage::Response {
            corr_id,
            ok,
            result,
            ..
        } = rx.recv().await.expect("an answer")
        else {
            panic!("a response");
        };
        assert_eq!((corr_id, ok), (9999, false));
        assert!(result.starts_with("quota"), "{result}");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_session_owns_only_its_own_tenant_and_users_processes() {
        let session = Session::new_inproc(
            8,
            install_state(0, true),
            mpsc::channel(1).0,
            "acme".to_string(),
            "alice".to_string(),
        );
        let owner = |tenant: &str, username: &str| process::Owner {
            tenant: tenant.to_string(),
            username: username.to_string(),
        };
        assert!(session.owns(&owner("acme", "alice")));
        assert!(!session.owns(&owner("acme", "bob")));
        assert!(!session.owns(&owner("other", "alice")));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_closed_session_ends_what_it_launched_and_only_detaches_what_it_attached() {
        let launched = ProcessId::new_v4();
        let attached = ProcessId::new_v4();
        let background = ProcessId::new_v4();
        let seen_launched = probe::spawn(launched);
        let seen_attached = probe::spawn(attached);
        let mut seen_background = probe::spawn(background);

        let mut session = Session::new_inproc(
            7,
            install_state(0, true),
            mpsc::channel(1).0,
            DEFAULT_TENANT.to_string(),
            DEFAULT_USER.to_string(),
        );
        session.attached_processes = vec![launched, attached];
        session.launched_processes.insert(launched);
        drop(session);

        match seen_launched.await {
            Ok(Seen::Terminate(Err(why))) => assert_eq!(why, "client disconnected"),
            other => panic!("a launched process must be terminated, got {other:?}"),
        }
        match seen_attached.await {
            Ok(Seen::Detach) => {}
            other => panic!("a merely attached process must only detach, got {other:?}"),
        }
        for _ in 0..8 {
            crate::rt::yield_now().await;
        }
        assert!(
            seen_background.try_recv().is_err(),
            "a process launched without outputs is not this session's to touch"
        );
        process::terminate(attached, Err(String::new()));
        process::terminate(background, Err(String::new()));
    }
}
