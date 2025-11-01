use crate::crypto::ParsedPrivateKey;
use crate::message::{
    CHUNK_SIZE_BYTES, ClientMessage, EventCode, QUERY_PROGRAM_EXISTS, ServerMessage,
};
use crate::utils::IdPool;
use anyhow::{Context, Result};
use base64::Engine;
use bytes::Bytes;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use rmp_serde::{decode, encode};
use std::sync::Arc;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use uuid::Uuid;

type InstanceId = Uuid;
type CorrId = u32;

/// A binary blob or a text-based event from an instance.
#[derive(Debug)]
pub enum InstanceEvent {
    /// An event from the server, like program completion or an error.
    Event { code: EventCode, message: String },
    /// A binary blob of data sent from the instance.
    Blob(Vec<u8>),
}

/// Holds the state for a blob being downloaded from the server.
#[derive(Debug)]
struct DownloadState {
    instance_id: InstanceId,
    buffer: Vec<u8>,
}

/// A client that interacts with the server.
pub struct Client {
    inner: Arc<ClientInner>,
    _server_event_rx: mpsc::Receiver<String>, // Keep the channel endpoint
    reader_handle: task::JoinHandle<()>,
    writer_handle: task::JoinHandle<()>,
}

/// State shared between the Client and its Instances.
#[derive(Debug)]
struct ClientInner {
    ws_writer_tx: UnboundedSender<Message>,
    corr_id_pool: Mutex<IdPool<CorrId>>,
    pending_requests: DashMap<CorrId, oneshot::Sender<(bool, String)>>,
    inst_event_tx: DashMap<InstanceId, mpsc::Sender<InstanceEvent>>,
    // Use a Mutex per entry to avoid deadlocking the DashMap shard
    pending_downloads: DashMap<String, Mutex<DownloadState>>, // Key: blob_hash
}

/// Represents a running program instance on the server.
#[derive(Debug)]
pub struct Instance {
    id: InstanceId,
    inner: Arc<ClientInner>,
    event_rx: mpsc::Receiver<InstanceEvent>,
}

/// Computes the blake3 hash for a slice of bytes.
pub fn hash_blob(blob: &[u8]) -> String {
    blake3::hash(blob).to_hex().to_string()
}

impl Instance {
    pub fn id(&self) -> InstanceId {
        self.id
    }

    /// Sends a string message to the instance (fire-and-forget).
    pub async fn send<T: ToString>(&self, message: T) -> Result<()> {
        let msg = ClientMessage::SignalInstance {
            instance_id: self.id.to_string(),
            message: message.to_string(),
        };
        self.inner
            .ws_writer_tx
            .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        Ok(())
    }

    /// Uploads a binary blob to the instance, handling chunking and awaiting confirmation.
    pub async fn upload_blob(&self, blob: &[u8]) -> Result<()> {
        let blob_hash = hash_blob(blob);
        let corr_id = self.inner.corr_id_pool.lock().await.acquire()?;
        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(corr_id, tx);

        let total_size = blob.len();
        // An empty blob is sent as one empty chunk.
        let total_chunks = if total_size == 0 {
            1
        } else {
            total_size.div_ceil(CHUNK_SIZE_BYTES)
        };

        for chunk_index in 0..total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let msg = ClientMessage::UploadBlob {
                corr_id,
                instance_id: self.id.to_string(),
                blob_hash: blob_hash.clone(),
                chunk_index,
                total_chunks,
                chunk_data: blob[start..end].to_vec(),
            };
            self.inner
                .ws_writer_tx
                .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        }

        let (successful, result) = rx.await?;
        self.inner.corr_id_pool.lock().await.release(corr_id)?;
        if successful {
            Ok(())
        } else {
            anyhow::bail!("Blob upload failed: {}", result)
        }
    }

    /// Receives the next event or blob from the instance.
    pub async fn recv(&mut self) -> Result<InstanceEvent> {
        self.event_rx
            .recv()
            .await
            .ok_or_else(|| anyhow::anyhow!("Event channel closed"))
    }

    /// Requests the server to terminate the instance (fire-and-forget).
    pub async fn terminate(&self) -> Result<()> {
        let msg = ClientMessage::TerminateInstance {
            instance_id: self.id.to_string(),
        };
        self.inner
            .ws_writer_tx
            .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        Ok(())
    }
}

impl Client {
    pub async fn connect(ws_host: &str) -> Result<Client> {
        let (ws_stream, _) = connect_async(ws_host).await?;
        let (mut ws_write, mut ws_read) = ws_stream.split();
        let (ws_writer_tx, mut ws_writer_rx) = unbounded_channel();
        let (server_event_tx, server_event_rx) = mpsc::channel(64);

        let inner = Arc::new(ClientInner {
            ws_writer_tx: ws_writer_tx.clone(),
            corr_id_pool: Mutex::new(IdPool::new(CorrId::MAX)),
            pending_requests: DashMap::new(),
            inst_event_tx: DashMap::new(),
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
                            handle_server_message(server_msg, &reader_inner, &server_event_tx)
                                .await;
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
            _server_event_rx: server_event_rx,
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

    async fn send_msg_and_wait(&self, mut msg: ClientMessage) -> Result<(bool, String)> {
        let corr_id_new = self.inner.corr_id_pool.lock().await.acquire()?;
        let corr_id_ref = match &mut msg {
            ClientMessage::Authenticate { corr_id, .. }
            | ClientMessage::Signature { corr_id, .. }
            | ClientMessage::InternalAuthenticate { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::LaunchInstance { corr_id, .. }
            | ClientMessage::Ping { corr_id } => corr_id,
            _ => anyhow::bail!("Invalid message type for this helper"),
        };
        *corr_id_ref = corr_id_new;

        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(corr_id_new, tx);
        self.inner
            .ws_writer_tx
            .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;

        let (successful, result) = rx.await?;
        self.inner.corr_id_pool.lock().await.release(corr_id_new)?;
        Ok((successful, result))
    }

    /// Authenticates the client with the server using a username and private key.
    pub async fn authenticate(&self, username: &str, private_key: &ParsedPrivateKey) -> Result<()> {
        // Send the authentication request to the server to get a challenge
        let msg = ClientMessage::Authenticate {
            corr_id: 0,
            username: username.to_string(),
        };
        let (successful, result) = self.send_msg_and_wait(msg).await?;

        if !successful {
            anyhow::bail!(
                "Authentication failed for username {}: {}",
                username,
                result
            )
        }

        // If the server has disabled authentication, we can return early.
        if result == "Already authenticated" {
            return Ok(());
        }

        // Otherwise, the server has enabled authentication and we need to sign
        // the challenge encoded in base64 with the private key.
        let challenge = base64::engine::general_purpose::STANDARD
            .decode(result.as_bytes())
            .context("Failed to decode challenge from base64")?;

        // Sign the challenge with the private key
        let signature_bytes = private_key.sign(&challenge)?;

        // Encode the signature to base64
        let signature = base64::engine::general_purpose::STANDARD.encode(&signature_bytes);

        // Send the signature to the server to verify the authentication
        let msg = ClientMessage::Signature {
            corr_id: 0,
            signature,
        };

        let (successful, result) = self.send_msg_and_wait(msg).await?;
        if successful {
            Ok(())
        } else {
            anyhow::bail!(
                "Signature verification failed for username {}: {}",
                username,
                result
            )
        }
    }

    /// Authenticates the client with the server using an internal token.
    /// This method is used for internal communication between the backend and the engine
    /// as well as between the Pie shell and the engine. This is not used for user authentication.
    pub async fn internal_authenticate(&self, token: &str) -> Result<()> {
        let msg = ClientMessage::InternalAuthenticate {
            corr_id: 0,
            token: token.to_string(),
        };
        let (successful, result) = self.send_msg_and_wait(msg).await?;
        if successful {
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
        let (successful, result) = self.send_msg_and_wait(msg).await?;
        if successful {
            Ok(result)
        } else {
            anyhow::bail!("Query failed: {}", result)
        }
    }

    pub async fn program_exists(&self, program_hash: &str) -> Result<bool> {
        self.query(QUERY_PROGRAM_EXISTS, program_hash.to_string())
            .await
            .map(|r| r == "true")
    }

    pub async fn upload_program(&self, blob: &[u8]) -> Result<()> {
        let program_hash = hash_blob(blob);
        let corr_id = self.inner.corr_id_pool.lock().await.acquire()?;
        let (tx, rx) = oneshot::channel();
        self.inner.pending_requests.insert(corr_id, tx);

        let total_size = blob.len();
        let total_chunks = if total_size == 0 {
            1
        } else {
            total_size.div_ceil(CHUNK_SIZE_BYTES)
        };

        for chunk_index in 0..total_chunks {
            let start = chunk_index * CHUNK_SIZE_BYTES;
            let end = (start + CHUNK_SIZE_BYTES).min(total_size);
            let msg = ClientMessage::UploadProgram {
                corr_id,
                program_hash: program_hash.clone(),
                chunk_index,
                total_chunks,
                chunk_data: blob[start..end].to_vec(),
            };
            self.inner
                .ws_writer_tx
                .send(Message::Binary(Bytes::from(encode::to_vec_named(&msg)?)))?;
        }

        let (successful, result) = rx.await?;
        self.inner.corr_id_pool.lock().await.release(corr_id)?;
        if successful {
            Ok(())
        } else {
            anyhow::bail!("Program upload failed: {}", result)
        }
    }

    pub async fn launch_instance(
        &self,
        program_hash: &str,
        arguments: Vec<String>,
    ) -> Result<Instance> {
        let msg = ClientMessage::LaunchInstance {
            corr_id: 0,
            program_hash: program_hash.to_string(),
            arguments,
        };
        let (successful, result) = self.send_msg_and_wait(msg).await?;
        if successful {
            let inst_id = Uuid::parse_str(&result)?;
            let (tx, rx) = mpsc::channel(64);
            self.inner.inst_event_tx.insert(inst_id, tx);
            Ok(Instance {
                id: inst_id,
                inner: Arc::clone(&self.inner),
                event_rx: rx,
            })
        } else {
            anyhow::bail!("Launch instance failed: {}", result)
        }
    }

    pub async fn ping(&self) -> Result<()> {
        let msg = ClientMessage::Ping { corr_id: 0 };
        let (successful, result) = self.send_msg_and_wait(msg).await?;
        if successful {
            Ok(())
        } else {
            anyhow::bail!("Ping failed: {}", result)
        }
    }
}

/// Main message handler function called by the reader task.
async fn handle_server_message(
    msg: ServerMessage,
    inner: &Arc<ClientInner>,
    server_event_tx: &mpsc::Sender<String>,
) {
    match msg {
        ServerMessage::Response {
            corr_id,
            successful,
            result,
        } => {
            if let Some((_, sender)) = inner.pending_requests.remove(&corr_id) {
                sender.send((successful, result)).ok();
            }
        }
        ServerMessage::InstanceEvent {
            instance_id,
            event,
            message,
        } => {
            if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
                if let Some(sender) = inner.inst_event_tx.get(&inst_id) {
                    sender
                        .send(InstanceEvent::Event {
                            code: EventCode::from_u32(event).unwrap(),
                            message,
                        })
                        .await
                        .ok();
                }
            }
        }
        ServerMessage::DownloadBlob {
            instance_id,
            blob_hash,
            chunk_index,
            total_chunks,
            chunk_data,
            ..
        } => {
            if !inner.pending_downloads.contains_key(&blob_hash) {
                if let Ok(id) = Uuid::parse_str(&instance_id) {
                    let state = DownloadState {
                        instance_id: id,
                        buffer: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                    };
                    inner
                        .pending_downloads
                        .insert(blob_hash.clone(), Mutex::new(state));
                }
            }
            if let Some(state_mutex) = inner.pending_downloads.get(&blob_hash) {
                let mut state = state_mutex.lock().await;
                state.buffer.extend_from_slice(&chunk_data);

                if chunk_index == total_chunks - 1 {
                    if let Some((_, state_mutex)) = inner.pending_downloads.remove(&blob_hash) {
                        let final_state = state_mutex.into_inner();
                        if hash_blob(&final_state.buffer) == blob_hash {
                            if let Some(sender) = inner.inst_event_tx.get(&final_state.instance_id)
                            {
                                sender
                                    .send(InstanceEvent::Blob(final_state.buffer))
                                    .await
                                    .ok();
                            }
                        }
                    }
                }
            }
        }
        ServerMessage::ServerEvent { message } => {
            server_event_tx.send(message).await.ok();
        }
        ServerMessage::Challenge { corr_id, challenge } => {
            if let Some((_, sender)) = inner.pending_requests.remove(&corr_id) {
                sender.send((true, challenge)).ok();
            }
        }
    }
}

/// When the server teminates, clear all pending requests, instance events, and downloads
/// to avoid locking up the client indefinitely.
async fn handle_server_termination(inner: &Arc<ClientInner>) {
    inner.pending_requests.clear();
    inner.inst_event_tx.clear();
    inner.pending_downloads.clear();
}
