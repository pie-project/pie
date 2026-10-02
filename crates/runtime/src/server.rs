mod data_transfer;
mod handler;
pub(crate) mod inbox;

pub use data_transfer::InFlightUpload;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;

use anyhow::{Result, anyhow};
use bytes::Bytes;
use client_api::message::{ClientMessage, ServerMessage as WireServerMessage};
use dashmap::DashMap;
use tokio::sync::{Mutex as TokioMutex, mpsc};

use crate::inferlet::process;
use crate::inferlet::{ProcessEvent, ProcessId, ProgramName};
use crate::service::{ServiceHandler, ServiceMap};

pub type ClientId = u32;

static STATE: OnceLock<Arc<ServerState>> = OnceLock::new();
static SESSION_OUTBOX: LazyLock<
    DashMap<ClientId, Arc<TokioMutex<mpsc::Receiver<WireServerMessage>>>>,
> = LazyLock::new(DashMap::new);

fn install_state(max_upload_bytes: usize) -> Arc<ServerState> {
    if let Some(state) = STATE.get() {
        return Arc::clone(state);
    }

    let state = Arc::new(ServerState {
        next_client_id: AtomicU32::new(1),
        max_upload_bytes,
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

pub(crate) fn init(max_upload_bytes: usize) {
    let _ = install_state(max_upload_bytes);
    inbox::spawn();
}

pub fn open_session() -> Result<ClientId> {
    let state = get_state()?;
    let id = state.next_client_id.fetch_add(1, Ordering::Relaxed);

    let (out_tx, out_rx) = mpsc::channel(1000);
    SESSION_OUTBOX.insert(id, Arc::new(TokioMutex::new(out_rx)));

    let session = Session::new_inproc(id, state, out_tx);
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

#[allow(dead_code)]
pub(crate) fn exists(client_id: ClientId) -> bool {
    CLIENT_SERVICES.contains(&client_id)
}

struct ServerState {
    next_client_id: AtomicU32,
    pub(super) max_upload_bytes: usize,
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
    state: Arc<ServerState>,
    pub(super) inflight_uploads: DashMap<UploadKey, InFlightUpload>,
    pub(super) attached_processes: Vec<ProcessId>,
    /// The subset of `attached_processes` this session launched with output
    /// capture — the launcher owns them, so its close ends them (below);
    /// processes merely attached to are another session's and only detach.
    pub(super) launched_processes: HashSet<ProcessId>,
    pub(super) installed_programs: HashSet<ProgramName>,
    pub(super) file_waiters: HashMap<ProcessId, tokio::sync::oneshot::Sender<Bytes>>,
    out_tx: mpsc::Sender<WireServerMessage>,
}

impl Session {
    fn new_inproc(
        id: ClientId,
        state: Arc<ServerState>,
        out_tx: mpsc::Sender<WireServerMessage>,
    ) -> Self {
        Session {
            id,
            username: "internal".to_string(),
            state,
            inflight_uploads: DashMap::new(),
            attached_processes: Vec::new(),
            launched_processes: HashSet::new(),
            installed_programs: HashSet::new(),
            file_waiters: HashMap::new(),
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
                self.handle_client_message(client_msg).await;
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
        let uuid_str = process_id.to_string();
        self.send(WireServerMessage::ProcessEvent {
            process_id: uuid_str,
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

            ClientMessage::CheckProgram {
                corr_id,
                name,
                version,
            } => self.handle_check_program(corr_id, name, version).await,

            ClientMessage::Query {
                corr_id,
                subject,
                record,
            } => self.handle_query(corr_id, subject, record).await,

            ClientMessage::AddProgram {
                corr_id,
                program_hash,
                file,
                version,
                force_overwrite,
                chunk_index,
                total_chunks,
                chunk_data,
            } => {
                self.handle_add_program(
                    corr_id,
                    program_hash,
                    file,
                    version,
                    force_overwrite,
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
            } => {
                self.handle_launch_process(corr_id, inferlet, input, capture_outputs)
                    .await
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

    #[tokio::test(flavor = "current_thread")]
    async fn a_closed_session_ends_what_it_launched_and_only_detaches_what_it_attached() {
        let launched = ProcessId::new_v4();
        let attached = ProcessId::new_v4();
        let background = ProcessId::new_v4();
        let seen_launched = probe::spawn(launched);
        let seen_attached = probe::spawn(attached);
        let mut seen_background = probe::spawn(background);

        let mut session = Session::new_inproc(7, install_state(0), mpsc::channel(1).0);
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
