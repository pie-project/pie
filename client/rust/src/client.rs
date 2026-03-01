use crate::crypto::ParsedPrivateKey;
use crate::message::{CHUNK_SIZE_BYTES, ClientMessage, ServerMessage};
use crate::utils::IdPool;
use anyhow::{Context, Result, anyhow};
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use rmp_serde::{decode, encode};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use uuid::Uuid;

type CorrId = u32;

/// Events received from a running process.
#[derive(Debug)]
pub enum ProcessEvent {
    /// Stdout output from the process.
    Stdout(String),
    /// Stderr output from the process.
    Stderr(String),
    /// An inferlet text message (via messaging::send).
    Message(String),
    /// A binary file sent from the inferlet.
    File(Vec<u8>),
    /// Process completed successfully with a return value.
    Return(String),
    /// Process terminated with an error.
    Error(String),
}

/// Holds the state for a file being downloaded from the server.
#[derive(Debug)]
struct DownloadState {
    process_id: String,
    buffer: Vec<u8>,
}

/// A client that interacts with the server.
pub struct Client {
    inner: Arc<ClientInner>,
    reader_handle: task::JoinHandle<()>,
    writer_handle: task::JoinHandle<()>,
}

/// State shared between the Client and its Processes.
#[derive(Debug)]
struct ClientInner {
    ws_writer_tx: UnboundedSender<Message>,
    corr_id_pool: IdPool<CorrId>,
    /// Single pending-request map: all request/reply commands use this.
    pending_requests: DashMap<CorrId, oneshot::Sender<(bool, String)>>,
    /// Per-process event channels.
    process_event_tx: DashMap<String, mpsc::Sender<ProcessEvent>>,
    /// In-flight file downloads (key: file_hash).
    pending_downloads: DashMap<String, Mutex<DownloadState>>,
}

/// Represents a running process on the server.
#[derive(Debug)]
pub struct Process {
    id: String,
    inner: Arc<ClientInner>,
    event_rx: mpsc::Receiver<ProcessEvent>,
}

/// Computes the blake3 hash for a slice of bytes.
pub fn hash_blob(blob: &[u8]) -> String {
    blake3::hash(blob).to_hex().to_string()
}

impl Process {
    /// Returns the process UUID string.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Sends a string message to the process (fire-and-forget).
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

    /// Uploads a binary file to the process (fire-and-forget, chunked).
    pub async fn transfer_file(&self, blob: &[u8]) -> Result<()> {
        let file_hash = hash_blob(blob);
        let total_size = blob.len();
        let total_chunks = if total_size == 0 { 1 } else { total_size.div_ceil(CHUNK_SIZE_BYTES) };

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

    /// Receives the next event from the process. Blocks until one is available.
    pub async fn recv(&mut self) -> Result<ProcessEvent> {
        self.event_rx
            .recv()
            .await
            .ok_or(anyhow!("Event channel closed"))
    }

    /// Non-blocking receive. Returns None if no event is available.
    pub fn try_recv(&mut self) -> Result<Option<ProcessEvent>> {
        match self.event_rx.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(mpsc::error::TryRecvError::Disconnected) => Err(anyhow!("Event channel closed")),
        }
    }
}

impl Client {
    pub async fn connect(ws_host: &str) -> Result<Client> {
        let (ws_stream, _) = connect_async(ws_host).await?;
        let (mut ws_write, mut ws_read) = ws_stream.split();
        let (ws_writer_tx, mut ws_writer_rx) = unbounded_channel();

        let inner = Arc::new(ClientInner {
            ws_writer_tx: ws_writer_tx.clone(),
            corr_id_pool: IdPool::new(CorrId::MAX),
            pending_requests: DashMap::new(),
            process_event_tx: DashMap::new(),
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

    /// Close the connection and clean up background tasks.
    pub async fn close(self) -> Result<()> {
        self.writer_handle.await?;
        self.reader_handle.abort();
        Ok(())
    }

    /// Send a command and wait for a Response { corr_id, ok, result }.
    async fn send_msg_and_wait(&self, mut msg: ClientMessage) -> Result<(bool, String)> {
        let corr_id_guard = self.inner.corr_id_pool.acquire().await?;
        let corr_id_ref = match &mut msg {
            ClientMessage::AuthIdentify { corr_id, .. }
            | ClientMessage::AuthProve { corr_id, .. }
            | ClientMessage::AuthByToken { corr_id, .. }
            | ClientMessage::CheckProgram { corr_id, .. }
            | ClientMessage::TerminateProcess { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::AddProgram { corr_id, .. }
            | ClientMessage::LaunchProcess { corr_id, .. }
            | ClientMessage::LaunchDaemon { corr_id, .. }
            | ClientMessage::ListProcesses { corr_id }
            | ClientMessage::RegisterMcpServer { corr_id, .. }
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

    /// Authenticates the client with the server using a username and private key.
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

        // If the engine has disabled public key authentication, we can return early.
        if result == "Authenticated (Engine disabled authentication)" {
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

    /// Authenticates the client with the server using an internal token.
    pub async fn auth_by_token(&self, token: &str) -> Result<()> {
        let msg = ClientMessage::AuthByToken {
            corr_id: 0,
            token: token.to_string(),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(())
        } else {
            anyhow::bail!("Internal authentication failed: {}", result)
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

    /// Check if a program exists on the server.
    ///
    /// The `inferlet` must be in `name@version` format (e.g., "text-completion@0.1.0").
    pub async fn check_program(
        &self,
        inferlet: &str,
        wasm_path: Option<&Path>,
        manifest_path: Option<&Path>,
    ) -> Result<bool> {
        use std::sync::LazyLock;
        use regex::Regex;

        static RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r"^([a-zA-Z0-9][a-zA-Z0-9_-]*)@(\d+\.\d+\.\d+)$").unwrap()
        });

        let caps = RE.captures(inferlet)
            .ok_or_else(|| anyhow!(
                "Invalid program identifier '{}': expected 'name@major.minor.patch'", inferlet
            ))?;
        let name = caps[1].to_string();
        let version = caps[2].to_string();

        let (wasm_hash, manifest_hash) = match (wasm_path, manifest_path) {
            (Some(wasm_p), Some(manifest_p)) => {
                let wasm_bytes = fs::read(wasm_p)
                    .with_context(|| format!("Failed to read WASM file: {:?}", wasm_p))?;
                let manifest_content = fs::read_to_string(manifest_p)
                    .with_context(|| format!("Failed to read manifest file: {:?}", manifest_p))?;
                (Some(hash_blob(&wasm_bytes)), Some(hash_blob(manifest_content.as_bytes())))
            }
            (None, None) => (None, None),
            _ => anyhow::bail!("wasm_path and manifest_path must both be provided or both be None"),
        };

        let msg = ClientMessage::CheckProgram {
            corr_id: 0,
            name,
            version,
            wasm_hash,
            manifest_hash,
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(result == "true")
        } else {
            anyhow::bail!("CheckProgram failed: {}", result)
        }
    }

    /// For backward compatibility. Delegates to `check_program`.
    pub async fn program_exists(
        &self,
        inferlet: &str,
        wasm_path: Option<&Path>,
        manifest_path: Option<&Path>,
    ) -> Result<bool> {
        self.check_program(inferlet, wasm_path, manifest_path).await
    }

    /// Upload a program to the server.
    pub async fn add_program(&self, wasm_path: &Path, manifest_path: &Path, force_overwrite: bool) -> Result<()> {
        let blob = fs::read(wasm_path)
            .with_context(|| format!("Failed to read WASM file: {:?}", wasm_path))?;
        let manifest = fs::read_to_string(manifest_path)
            .with_context(|| format!("Failed to read manifest file: {:?}", manifest_path))?;

        let program_hash = hash_blob(&blob);
        let corr_id_guard = self.inner.corr_id_pool.acquire().await?;
        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(*corr_id_guard, tx);

        let total_size = blob.len();
        let total_chunks = if total_size == 0 { 1 } else { total_size.div_ceil(CHUNK_SIZE_BYTES) };

        for chunk_index in 0..total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let msg = ClientMessage::AddProgram {
                corr_id: *corr_id_guard,
                program_hash: program_hash.clone(),
                manifest: manifest.to_string(),
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
            Ok(())
        } else {
            anyhow::bail!("Program install failed: {}", result)
        }
    }

    /// Launches an instance of a program. Returns a `Process` for interaction.
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

        // result is the UUID string
        let process_id = result;
        let (tx, rx) = mpsc::channel(64);
        self.inner.process_event_tx.insert(process_id.clone(), tx);

        Ok(Process {
            id: process_id,
            inner: Arc::clone(&self.inner),
            event_rx: rx,
        })
    }

    pub async fn attach_process(&self, process_id: &str) -> Result<Process> {
        // Validate UUID format
        let _uuid = Uuid::parse_str(process_id)?;
        let msg = ClientMessage::AttachProcess {
            corr_id: 0,
            process_id: process_id.to_string(),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;

        if !ok {
            anyhow::bail!("Attach process failed: {}", result);
        }

        let (tx, rx) = mpsc::channel(64);
        self.inner.process_event_tx.insert(process_id.to_string(), tx);

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

    /// List running processes. Returns a list of process UUID strings.
    pub async fn list_processes(&self) -> Result<Vec<String>> {
        let msg = ClientMessage::ListProcesses { corr_id: 0 };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            let ids: Vec<String> = result.split(',')
                .map(|s| s.trim().trim_matches('"').trim_matches('[').trim_matches(']').to_string())
                .filter(|s| !s.is_empty())
                .collect();
            Ok(ids)
        } else {
            anyhow::bail!("List processes failed: {}", result)
        }
    }

    /// Terminates a process by its UUID string.
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

    /// Registers an MCP server for this session.
    /// All inferlets launched in this session can discover and connect to it.
    pub async fn register_mcp_server(
        &self,
        name: &str,
        transport: &str,
        command: Option<&str>,
        args: Option<Vec<String>>,
        url: Option<&str>,
    ) -> Result<()> {
        let msg = ClientMessage::RegisterMcpServer {
            corr_id: 0,
            name: name.to_string(),
            transport: transport.to_string(),
            command: command.map(|s| s.to_string()),
            args,
            url: url.map(|s| s.to_string()),
        };
        let (ok, result) = self.send_msg_and_wait(msg).await?;
        if ok {
            Ok(())
        } else {
            anyhow::bail!("Register MCP server failed: {}", result)
        }
    }
}

// =============================================================================
// Server Message Handler
// =============================================================================

/// Routes incoming server messages to the appropriate handler.
async fn handle_server_message(
    msg: ServerMessage,
    inner: &Arc<ClientInner>,
) {
    match msg {
        ServerMessage::Response { corr_id, ok, result } => {
            if let Some((_, sender)) = inner.pending_requests.remove(&corr_id) {
                sender.send((ok, result)).ok();
            }
        }
        ServerMessage::ProcessEvent { process_id, event, value } => {
            if let Some(sender) = inner.process_event_tx.get(&process_id) {
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
                sender.send(process_event).await.ok();

                // Clean up event channel on terminal events
                if event == "return" || event == "error" {
                    drop(sender);
                    inner.process_event_tx.remove(&process_id);
                }
            }
        }
        ServerMessage::File {
            process_id,
            file_hash,
            chunk_index,
            total_chunks,
            chunk_data,
        } => {
            // Initialize download state on first chunk
            if !inner.pending_downloads.contains_key(&file_hash) {
                let state = DownloadState {
                    process_id: process_id.clone(),
                    buffer: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                };
                inner
                    .pending_downloads
                    .insert(file_hash.clone(), Mutex::new(state));
            }

            // Accumulate chunk data, then drop all guards before any remove().
            // SAFETY: We must drop the DashMap Ref guard before calling .remove(),
            // because .get() holds a shard read-lock and .remove() needs a write-lock
            // on the same shard — holding both would deadlock.
            let is_last = chunk_index == total_chunks - 1;
            if let Some(state_mutex) = inner.pending_downloads.get(&file_hash) {
                let mut state = state_mutex.lock().await;
                state.buffer.extend_from_slice(&chunk_data);
                drop(state);        // release Mutex guard
            }
            // DashMap Ref dropped here (end of `if let` scope)

            // Finalize on last chunk — no guards held
            if is_last {
                if let Some((_, state_mutex)) = inner.pending_downloads.remove(&file_hash) {
                    let final_state = state_mutex.into_inner();
                    if hash_blob(&final_state.buffer) == file_hash {
                        if let Some(sender) = inner.process_event_tx.get(&final_state.process_id) {
                            sender
                                .send(ProcessEvent::File(final_state.buffer))
                                .await
                                .ok();
                        }
                    }
                }
            }
        }
        ServerMessage::McpRequest {
            corr_id,
            process_id: _,
            server_name,
            method,
            params,
        } => {
            // TODO: Forward to local MCP server and send real response.
            // For now, return a stub error so the protocol is wired.
            eprintln!(
                "MCP request received: server={}, method={}, params={}",
                server_name, method, params
            );
            let response = ClientMessage::McpResponse {
                corr_id,
                ok: false,
                result: "Client-side MCP bridge not yet implemented".to_string(),
            };
            if let Ok(encoded) = rmp_serde::encode::to_vec_named(&response) {
                inner
                    .ws_writer_tx
                    .send(Message::Binary(Bytes::from(encoded)))
                    .ok();
            }
        }
    }
}

/// When the server terminates, clear all pending state.
async fn handle_server_termination(inner: &Arc<ClientInner>) {
    inner.pending_requests.clear();
    inner.process_event_tx.clear();
    inner.pending_downloads.clear();
}
