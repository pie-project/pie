use crate::crypto::ParsedPrivateKey;
use crate::message::{CHUNK_SIZE_BYTES, ClientMessage, ServerMessage};
use crate::utils::IdPool;
use anyhow::{Context, Result, anyhow};
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use rmp_serde::{decode, encode};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{client::IntoClientRequest, http::HeaderValue, protocol::Message},
};
use uuid::Uuid;

type CorrId = u32;

#[derive(Debug)]
pub enum ProcessEvent {
    Stdout(String),
    Stderr(String),
    Message(String),
    File(ReceivedFile),
    Return(String),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ReceivedFile {
    pub name: Option<String>,
    pub data: Vec<u8>,
}

impl ReceivedFile {
    pub fn file_name(&self, fallback: &str) -> String {
        let raw = self.name.as_deref().unwrap_or("");
        let tail = raw.rsplit(['/', '\\']).next().unwrap_or("");
        let stripped = tail.replace('\0', "");
        let base = stripped.trim();
        if base.is_empty() || base == "." || base == ".." {
            fallback.to_string()
        } else {
            base.to_string()
        }
    }
}

#[derive(Debug)]
enum ProcessEventRoute {
    Buffered(BufferedProcessEvents),
    Attached(mpsc::Sender<ProcessEvent>),
}

#[derive(Debug)]
struct BufferedProcessEvents {
    events: VecDeque<ProcessEvent>,
    updated_at: Instant,
    terminal: bool,
}

const MAX_BUFFERED_PROCESSES: usize = 1024;
const MAX_BUFFERED_EVENTS_PER_PROCESS: usize = 1024;
const BUFFERED_PROCESS_TTL: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct DownloadState {
    process_id: String,
    buffer: Vec<u8>,
    name: Option<String>,
}

pub struct Client {
    inner: Arc<ClientInner>,
    reader_handle: task::JoinHandle<()>,
    writer_handle: task::JoinHandle<()>,
}

#[derive(Debug)]
struct ClientInner {
    ws_writer_tx: UnboundedSender<Message>,
    corr_id_pool: IdPool<CorrId>,
    pending_requests: DashMap<CorrId, oneshot::Sender<(bool, String)>>,
    process_events: StdMutex<HashMap<String, ProcessEventRoute>>,
    pending_downloads: DashMap<String, Mutex<DownloadState>>,
}

#[derive(Debug)]
pub struct Process {
    id: String,
    inner: Arc<ClientInner>,
    event_rx: mpsc::Receiver<ProcessEvent>,
}

pub fn hash_blob(blob: &[u8]) -> String {
    blake3::hash(blob).to_hex().to_string()
}

impl Process {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub async fn signal<T: ToString>(&self, message: T) -> Result<()> {
        let msg = ClientMessage::SignalProcess {
            process_id: self.id.clone(),
            message: message.to_string(),
        };
        self.inner
            .ws_writer_tx
            .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        Ok(())
    }

    pub async fn transfer_file(&self, blob: &[u8]) -> Result<()> {
        let file_hash = hash_blob(blob);
        let total_size = blob.len();
        let total_chunks = if total_size == 0 {
            1
        } else {
            total_size.div_ceil(CHUNK_SIZE_BYTES)
        };

        for chunk_index in 0..total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let msg = ClientMessage::TransferFile {
                process_id: self.id.clone(),
                file_hash: file_hash.clone(),
                chunk_index,
                total_chunks,
                chunk_data: blob[start..end].to_vec(),
            };
            self.inner
                .ws_writer_tx
                .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        }
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<ProcessEvent> {
        self.event_rx
            .recv()
            .await
            .ok_or(anyhow!("Event channel closed"))
    }

    pub fn try_recv(&mut self) -> Result<Option<ProcessEvent>> {
        match self.event_rx.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(anyhow!("Event channel closed")),
        }
    }

    pub async fn wait_for_return(&mut self) -> Result<String> {
        loop {
            match self.recv().await? {
                ProcessEvent::Return(value) => return Ok(value),
                ProcessEvent::Error(e) => return Err(anyhow!("inferlet returned an error: {e}")),
                ProcessEvent::Stdout(s) | ProcessEvent::Stderr(s) => eprint!("{s}"),
                ProcessEvent::Message(_) | ProcessEvent::File(_) => {}
            }
        }
    }
}

impl Client {
    pub async fn connect(ws_host: &str) -> Result<Client> {
        Self::connect_inner(connect_async(ws_host).await?.0)
    }

    pub async fn connect_with_identity(ws_host: &str, identity: &str) -> Result<Client> {
        let mut request = ws_host.into_client_request()?;
        request
            .headers_mut()
            .insert("x-pie-identity", HeaderValue::from_str(identity)?);
        Self::connect_inner(connect_async(request).await?.0)
    }

    fn connect_inner(
        ws_stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> Result<Client> {
        let (mut ws_write, mut ws_read) = ws_stream.split();
        let (ws_writer_tx, mut ws_writer_rx) = unbounded_channel();

        let inner = Arc::new(ClientInner {
            ws_writer_tx: ws_writer_tx.clone(),
            corr_id_pool: IdPool::new(CorrId::MAX),
            pending_requests: DashMap::new(),
            process_events: StdMutex::new(HashMap::new()),
            pending_downloads: DashMap::new(),
        });

        let writer_handle = task::spawn(async move {
            while let Some(msg) = ws_writer_rx.recv().await {
                if ws_write.send(msg).await.is_err() {
                    break;
                }
            }
            let _ = ws_write.close().await;
        });

        let reader_inner = Arc::clone(&inner);
        let reader_handle = task::spawn(async move {
            while let Some(Ok(msg)) = ws_read.next().await {
                match msg {
                    Message::Binary(bin) => {
                        if let Ok(server_msg) = decode::from_slice::<ServerMessage>(&bin) {
                            handle_server_message(server_msg, &reader_inner).await;
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            handle_server_termination(&reader_inner).await;
        });

        Ok(Client {
            inner,
            reader_handle,
            writer_handle,
        })
    }

    pub async fn close(self) -> Result<()> {
        let Client {
            inner,
            reader_handle,
            writer_handle,
        } = self;
        reader_handle.abort();
        let _ = reader_handle.await;
        drop(inner);
        writer_handle.await?;
        Ok(())
    }

    async fn send_msg_and_wait(&self, mut msg: ClientMessage) -> Result<(bool, String)> {
        let corr_id_guard = self.inner.corr_id_pool.acquire().await?;
        let corr_id_ref = match &mut msg {
            ClientMessage::AuthIdentify { corr_id, .. }
            | ClientMessage::AuthProve { corr_id, .. }
            | ClientMessage::CheckProgram { corr_id, .. }
            | ClientMessage::TerminateProcess { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::AddProgram { corr_id, .. }
            | ClientMessage::LaunchProcess { corr_id, .. }
            | ClientMessage::ListProcesses { corr_id }
            | ClientMessage::Ping { corr_id } => corr_id,
            _ => anyhow::bail!("Invalid message type for this helper"),
        };
        *corr_id_ref = *corr_id_guard;

        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(*corr_id_guard, tx);
        self.inner
            .ws_writer_tx
            .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;

        let (ok, result) = rx.await?;
        Ok((ok, result))
    }

    pub async fn authenticate(
        &self,
        username: &str,
        private_key: &Option<ParsedPrivateKey>,
    ) -> Result<()> {
        let msg = ClientMessage::AuthIdentify {
            corr_id: 0,
            username: username.to_string(),
        };
        let (ok, result) = self
            .send_msg_and_wait(msg)
            .await
            .context("Failed to send identification message to engine")?;

        if !ok {
            anyhow::bail!("Username '{}' rejected by engine: {}", username, result)
        }

        if result == "Authenticated (Engine disabled authentication)"
            || result == "Already authenticated"
        {
            return Ok(());
        }

        let private_key = private_key
            .as_ref()
            .context("Client private key is required when engine uses public key authentication")?;

        let challenge = base64::engine::general_purpose::STANDARD
            .decode(result.as_bytes())
            .context("Failed to decode challenge from base64")?;

        let signature_bytes = private_key.sign(&challenge)?;
        let signature = base64::engine::general_purpose::STANDARD.encode(&signature_bytes);

        let msg = ClientMessage::AuthProve {
            corr_id: 0,
            signature,
        };

        let (ok, result) = self
            .send_msg_and_wait(msg)
            .await
            .context("Failed to send signature message to engine")?;
        if ok {
            Ok(())
        } else {
            anyhow::bail!(
                "Signature verification failed for username '{}': {}",
                username,
                result
            )
        }
    }

    pub async fn query<T: ToString>(&self, subject: T, record: String) -> Result<String> {
        let msg = ClientMessage::Query {
            corr_id: 0,
            subject: subject.to_string(),
            record,
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(result)
        } else {
            anyhow::bail!("Query failed: {}", result)
        }
    }

    /// Whether the server holds `name@version`.
    pub async fn check_program(&self, inferlet: &str) -> Result<bool> {
        use regex::Regex;
        use std::sync::LazyLock;

        static RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"^([a-zA-Z0-9][a-zA-Z0-9_-]*)@(\d+\.\d+\.\d+)$").unwrap());

        let caps = RE.captures(inferlet).ok_or_else(|| {
            anyhow!(
                "Invalid program identifier '{}': expected 'name@major.minor.patch'",
                inferlet
            )
        })?;
        let msg = ClientMessage::CheckProgram {
            corr_id: 0,
            name: caps[1].to_string(),
            version: caps[2].to_string(),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(result == "true")
        } else {
            anyhow::bail!("CheckProgram failed: {}", result)
        }
    }

    pub async fn add_program(
        &self,
        path: &Path,
        version: Option<&str>,
        force_overwrite: bool,
    ) -> Result<String> {
        let blob = fs::read(path).with_context(|| format!("Failed to read {:?}", path))?;
        let file = path
            .file_name()
            .and_then(|f| f.to_str())
            .with_context(|| format!("{:?} has no file name", path))?;
        self.add_program_bytes(&blob, file, version, force_overwrite)
            .await
    }

    pub async fn add_program_bytes(
        &self,
        blob: &[u8],
        file: &str,
        version: Option<&str>,
        force_overwrite: bool,
    ) -> Result<String> {
        let program_hash = hash_blob(blob);
        let corr_id_guard = self.inner.corr_id_pool.acquire().await?;
        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(*corr_id_guard, tx);

        let total_size = blob.len();
        let total_chunks = if total_size == 0 {
            1
        } else {
            total_size.div_ceil(CHUNK_SIZE_BYTES)
        };

        for chunk_index in 0..total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let msg = ClientMessage::AddProgram {
                corr_id: *corr_id_guard,
                program_hash: program_hash.clone(),
                file: file.to_string(),
                version: version.map(str::to_string),
                force_overwrite,
                chunk_index,
                total_chunks,
                chunk_data: blob[start..end].to_vec(),
            };
            self.inner
                .ws_writer_tx
                .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        }

        let (ok, result) = rx.await?;
        if ok {
            Ok(result)
        } else {
            anyhow::bail!("Program install failed: {}", result)
        }
    }

    pub async fn launch_process(
        &self,
        inferlet: String,
        input: String,
        capture_outputs: bool,
    ) -> Result<Process> {
        let msg = ClientMessage::LaunchProcess {
            corr_id: 0,
            inferlet,
            input,
            capture_outputs,
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;

        if !ok {
            anyhow::bail!("Launch process failed: {}", result);
        }

        let process_id = result;
        let rx = attach_process_events(&self.inner, &process_id);

        Ok(Process {
            id: process_id,
            inner: Arc::clone(&self.inner),
            event_rx: rx,
        })
    }

    pub async fn attach_process(&self, process_id: &str) -> Result<Process> {
        let _uuid = Uuid::parse_str(process_id)?;
        let msg = ClientMessage::AttachProcess {
            corr_id: 0,
            process_id: process_id.to_string(),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;

        if !ok {
            anyhow::bail!("Attach process failed: {}", result);
        }

        let rx = attach_process_events(&self.inner, process_id);

        Ok(Process {
            id: process_id.to_string(),
            inner: Arc::clone(&self.inner),
            event_rx: rx,
        })
    }

    pub async fn ping(&self) -> Result<()> {
        let msg = ClientMessage::Ping { corr_id: 0 };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(())
        } else {
            anyhow::bail!("Ping failed: {}", result)
        }
    }

    pub async fn list_processes(&self) -> Result<Vec<String>> {
        let msg = ClientMessage::ListProcesses { corr_id: 0 };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            let ids: Vec<String> = result
                .split(',')
                .map(|s| {
                    s.trim()
                        .trim_matches('"')
                        .trim_matches('[')
                        .trim_matches(']')
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .collect();
            Ok(ids)
        } else {
            anyhow::bail!("List processes failed: {}", result)
        }
    }

    pub async fn terminate_process(&self, process_id: &str) -> Result<()> {
        let msg = ClientMessage::TerminateProcess {
            corr_id: 0,
            process_id: process_id.to_string(),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(())
        } else {
            anyhow::bail!("Terminate process failed: {}", result)
        }
    }
}

async fn handle_server_message(msg: ServerMessage, inner: &Arc<ClientInner>) {
    match msg {
        ServerMessage::Response {
            corr_id,
            ok,
            result,
        } => {
            if let Some((_, sender)) = inner.pending_requests.remove(&corr_id) {
                sender.send((ok, result)).ok();
            }
        }
        ServerMessage::ProcessEvent {
            process_id,
            event,
            value,
        } => {
            let process_event = match event.as_str() {
                "stdout" => ProcessEvent::Stdout(value),
                "stderr" => ProcessEvent::Stderr(value),
                "message" => ProcessEvent::Message(value),
                "return" => ProcessEvent::Return(value),
                "error" => ProcessEvent::Error(value),
                _ => {
                    eprintln!("Unknown event type: {}", event);
                    return;
                }
            };
            route_process_event(inner, process_id, process_event).await;
        }
        ServerMessage::File {
            process_id,
            file_hash,
            chunk_index,
            total_chunks,
            chunk_data,
            name,
        } => {
            if !inner.pending_downloads.contains_key(&file_hash) {
                let state = DownloadState {
                    process_id: process_id.clone(),
                    buffer: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                    name: name.clone(),
                };
                inner
                    .pending_downloads
                    .insert(file_hash.clone(), Mutex::new(state));
            }

            let is_last = chunk_index == total_chunks - 1;
            if let Some(state_mutex) = inner.pending_downloads.get(&file_hash) {
                let mut state = state_mutex.lock().await;
                state.buffer.extend_from_slice(&chunk_data);
                drop(state);
            }

            if is_last && let Some((_, state_mutex)) = inner.pending_downloads.remove(&file_hash) {
                let final_state = state_mutex.into_inner();
                if hash_blob(&final_state.buffer) == file_hash {
                    route_process_event(
                        inner,
                        final_state.process_id,
                        ProcessEvent::File(ReceivedFile {
                            name: final_state.name,
                            data: final_state.buffer,
                        }),
                    )
                    .await;
                }
            }
        }
    }
}

async fn handle_server_termination(inner: &Arc<ClientInner>) {
    inner.pending_requests.clear();
    inner
        .process_events
        .lock()
        .expect("process event routes mutex poisoned")
        .clear();
    inner.pending_downloads.clear();
}

fn is_terminal_process_event(event: &ProcessEvent) -> bool {
    matches!(event, ProcessEvent::Return(_) | ProcessEvent::Error(_))
}

fn purge_expired_process_events(routes: &mut HashMap<String, ProcessEventRoute>) {
    let now = Instant::now();
    routes.retain(|process_id, route| {
        let keep = match route {
            ProcessEventRoute::Buffered(buffered) => {
                now.duration_since(buffered.updated_at) <= BUFFERED_PROCESS_TTL
            }
            ProcessEventRoute::Attached(_) => true,
        };
        if !keep {
            eprintln!("Discarding expired events for unattached process {process_id}");
        }
        keep
    });
}

fn attach_process_events(inner: &ClientInner, process_id: &str) -> mpsc::Receiver<ProcessEvent> {
    let mut routes = inner
        .process_events
        .lock()
        .expect("process event routes mutex poisoned");
    purge_expired_process_events(&mut routes);
    let buffered = match routes.remove(process_id) {
        Some(ProcessEventRoute::Buffered(buffered)) => buffered.events,
        Some(ProcessEventRoute::Attached(_)) | None => VecDeque::new(),
    };
    let terminal = buffered.iter().any(is_terminal_process_event);
    let (tx, rx) = mpsc::channel(buffered.len().max(64));
    for event in buffered {
        tx.try_send(event)
            .expect("new process event channel has capacity for buffered events");
    }
    if !terminal {
        routes.insert(process_id.to_string(), ProcessEventRoute::Attached(tx));
    }
    rx
}

async fn route_process_event(inner: &ClientInner, process_id: String, event: ProcessEvent) {
    let terminal = is_terminal_process_event(&event);
    let mut event = Some(event);
    let sender = {
        let mut routes = inner
            .process_events
            .lock()
            .expect("process event routes mutex poisoned");
        purge_expired_process_events(&mut routes);
        match routes.get_mut(&process_id) {
            Some(ProcessEventRoute::Attached(sender)) => {
                let sender = sender.clone();
                if terminal {
                    routes.remove(&process_id);
                }
                Some(sender)
            }
            Some(ProcessEventRoute::Buffered(buffered)) => {
                if buffered.terminal {
                    eprintln!("Discarding event received after terminal event for {process_id}");
                } else if buffered.events.len() >= MAX_BUFFERED_EVENTS_PER_PROCESS {
                    buffered.events.clear();
                    buffered.events.push_back(ProcessEvent::Error(
                        "process event buffer overflowed before attachment".to_string(),
                    ));
                    buffered.terminal = true;
                } else {
                    buffered
                        .events
                        .push_back(event.take().expect("process event present"));
                    buffered.terminal = terminal;
                }
                buffered.updated_at = Instant::now();
                None
            }
            None => {
                let buffered_count = routes
                    .values()
                    .filter(|route| matches!(route, ProcessEventRoute::Buffered(_)))
                    .count();
                if buffered_count >= MAX_BUFFERED_PROCESSES
                    && let Some(oldest) = routes
                        .iter()
                        .filter_map(|(id, route)| match route {
                            ProcessEventRoute::Buffered(buffered) => {
                                Some((id.clone(), buffered.updated_at))
                            }
                            ProcessEventRoute::Attached(_) => None,
                        })
                        .min_by_key(|(_, updated_at)| *updated_at)
                        .map(|(id, _)| id)
                {
                    routes.remove(&oldest);
                    eprintln!("Discarding oldest unattached process events for {oldest}");
                }
                routes.insert(
                    process_id,
                    ProcessEventRoute::Buffered(BufferedProcessEvents {
                        events: VecDeque::from([event.take().expect("process event present")]),
                        updated_at: Instant::now(),
                        terminal,
                    }),
                );
                None
            }
        }
    };
    if let Some(sender) = sender {
        sender
            .send(event.expect("attached process event present"))
            .await
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_inner() -> Arc<ClientInner> {
        let (ws_writer_tx, _) = unbounded_channel();
        Arc::new(ClientInner {
            ws_writer_tx,
            corr_id_pool: IdPool::new(CorrId::MAX),
            pending_requests: DashMap::new(),
            process_events: StdMutex::new(HashMap::new()),
            pending_downloads: DashMap::new(),
        })
    }

    #[tokio::test]
    async fn buffers_terminal_event_until_process_receiver_attaches() {
        let inner = test_inner();
        route_process_event(
            &inner,
            "fast-process".to_string(),
            ProcessEvent::Return("done".to_string()),
        )
        .await;

        let mut rx = attach_process_events(&inner, "fast-process");
        assert!(matches!(
            rx.recv().await,
            Some(ProcessEvent::Return(value)) if value == "done"
        ));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn preserves_events_buffered_before_process_receiver_attaches() {
        let inner = test_inner();
        route_process_event(
            &inner,
            "process".to_string(),
            ProcessEvent::Stdout("first".to_string()),
        )
        .await;
        let mut rx = attach_process_events(&inner, "process");
        route_process_event(
            &inner,
            "process".to_string(),
            ProcessEvent::Return("second".to_string()),
        )
        .await;

        assert!(matches!(
            rx.recv().await,
            Some(ProcessEvent::Stdout(value)) if value == "first"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(ProcessEvent::Return(value)) if value == "second"
        ));
        assert!(rx.recv().await.is_none());
    }
}
