//! Dummy backend implementation for testing purposes.
//!
//! This module provides a minimal backend that:
//! - Connects to the Pie engine via WebSocket
//! - Authenticates using internal authentication
//! - Registers as a remote service
//! - Responds to heartbeats via ZMQ
//! - Returns errors for all other handler requests
//!
//! The dummy backend is primarily used for CI testing to verify engine behavior
//! without requiring a full model backend.

use crate::model::request::{HANDSHAKE_ID, HEARTBEAT_ID, HandshakeRequest, HandshakeResponse};
use anyhow::{Context, Result};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use pie_client::message::{ClientMessage, ServerMessage};
use rand::Rng;
use rmp_serde::encode;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use tokio::sync::oneshot;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message as WsMessage};
use zeromq::{RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// Configuration for the dummy backend.
#[derive(Debug, Clone)]
pub struct DummyBackendConfig {
    pub controller_host: String,
    pub controller_port: u16,
    pub internal_auth_token: String,
}

/// Starts the dummy backend.
///
/// This function connects to the engine, authenticates, registers as a service,
/// and starts listening for ZMQ messages. It runs indefinitely until the task is cancelled.
pub async fn start_dummy_backend(config: DummyBackendConfig) -> Result<()> {
    println!("[Dummy Backend] Starting...");

    // Connect to the engine via WebSocket
    let controller_url = format!("ws://{}:{}", config.controller_host, config.controller_port);
    let (ws_stream, _) = connect_async(&controller_url)
        .await
        .context(format!("Failed to connect to engine at {}", controller_url))?;

    let (mut ws_write, mut ws_read) = ws_stream.split();

    // Authenticate with the engine
    let auth_msg = ClientMessage::InternalAuthenticate {
        corr_id: 0,
        token: config.internal_auth_token.clone(),
    };
    let auth_bytes = Bytes::from(
        encode::to_vec_named(&auth_msg).context("Failed to encode authentication message")?,
    );
    ws_write
        .send(WsMessage::Binary(auth_bytes))
        .await
        .context("Failed to send authentication message")?;

    // Wait for authentication response
    let response = ws_read
        .next()
        .await
        .context("Connection closed before authentication response")??;
    let response_data = response.into_data();
    let server_msg: ServerMessage = rmp_serde::from_slice(&response_data)
        .context("Failed to decode authentication response")?;

    match server_msg {
        ServerMessage::Response {
            successful, result, ..
        } => {
            if !successful {
                anyhow::bail!("Authentication failed: {}", result);
            }
        }
        _ => anyhow::bail!("Unexpected response type for authentication"),
    }
    println!("[Dummy Backend] Connected to engine at {}", controller_url);

    // Start ZMQ server before registering with the engine
    let (ready_tx, ready_rx) = oneshot::channel();
    let zmq_handle = tokio::spawn(run_zmq_server(ready_tx));

    // Wait for ZMQ server to be ready and get the endpoint
    let endpoint = ready_rx.await.context("ZMQ server failed to start")?;

    // Register with the engine
    let register_msg = ClientMessage::AttachRemoteService {
        corr_id: 0,
        endpoint,
        service_type: "model".to_string(),
        service_name: "dummy-model".to_string(),
    };
    let register_bytes = Bytes::from(
        encode::to_vec_named(&register_msg).context("Failed to encode registration message")?,
    );
    ws_write
        .send(WsMessage::Binary(register_bytes))
        .await
        .context("Failed to send registration message")?;

    // Wait for registration response
    let response = ws_read
        .next()
        .await
        .context("Connection closed before registration response")??;
    let response_data = response.into_data();
    let server_msg: ServerMessage =
        rmp_serde::from_slice(&response_data).context("Failed to decode registration response")?;

    match server_msg {
        ServerMessage::Response {
            successful, result, ..
        } => {
            if !successful {
                anyhow::bail!("Registration failed: {}", result);
            }
        }
        _ => anyhow::bail!("Unexpected response type for registration"),
    }
    println!("[Dummy Backend] Registered with engine");

    // Keep the WebSocket connection alive by reading messages
    // (though we don't expect any for the dummy backend)
    let ws_handle = tokio::spawn(async move {
        while let Some(result) = ws_read.next().await {
            match result {
                Ok(msg) => {
                    eprintln!("[Dummy Backend] Unexpected WebSocket message: {:?}", msg);
                }
                Err(e) => {
                    eprintln!("[Dummy Backend] WebSocket error: {}", e);
                    break;
                }
            }
        }
    });

    // Wait for either task to complete (or both)
    tokio::select! {
        result = zmq_handle => {
            match result {
                Ok(Ok(())) => println!("[Dummy Backend] ZMQ server stopped"),
                Ok(Err(e)) => eprintln!("[Dummy Backend] ZMQ server error: {}", e),
                Err(e) => eprintln!("[Dummy Backend] ZMQ task panicked: {}", e),
            }
        }
        _ = ws_handle => {
            println!("[Dummy Backend] WebSocket connection closed");
        }
    }

    Ok(())
}

/// Runs the ZMQ server loop.
///
/// This function listens for ZMQ messages and responds appropriately:
/// - Heartbeats: Responds with an empty success message
/// - All other handlers: Responds with an error message
async fn run_zmq_server(ready_tx: oneshot::Sender<String>) -> Result<()> {
    const MAX_RETRIES: u32 = 3;

    let mut socket = RouterSocket::new();
    let endpoint: String;

    // Try to find an available IPC endpoint, retry up to MAX_RETRIES times
    let mut attempt = 0;
    loop {
        attempt += 1;
        let unique_id: u32 = rand::rng().random_range(100000..=999999);
        let candidate_endpoint = format!("ipc:///tmp/pie-service-{}", unique_id);
        let socket_path = format!("/tmp/pie-service-{}", unique_id);

        // Check if the socket file already exists
        if !Path::new(&socket_path).exists() {
            // Endpoint is available, try to bind
            endpoint = candidate_endpoint;
            socket
                .bind(&endpoint)
                .await
                .context(format!("Failed to bind ZMQ socket to {}", endpoint))?;
            println!("[Dummy Backend] ZMQ server listening on {}", endpoint);

            // Signal that the server is ready and send the endpoint
            let _ = ready_tx.send(endpoint);
            break;
        }

        if attempt >= MAX_RETRIES {
            anyhow::bail!(
                "Failed to find available IPC endpoint after {} attempts",
                MAX_RETRIES
            );
        }
    }

    loop {
        // Receive multipart message
        let zmq_msg = socket
            .recv()
            .await
            .context("Failed to receive ZMQ message")?;

        let mut frames = zmq_msg.into_vecdeque();

        if frames.len() < 3 {
            eprintln!(
                "[Dummy Backend] Received invalid message with {} frames",
                frames.len()
            );
            continue;
        }

        // Extract message components: [client_identity, corr_id_bytes, handler_id_bytes, ...payloads]
        let client_identity = frames.pop_front().unwrap();
        let corr_id_bytes = frames.pop_front().unwrap();
        let handler_id_bytes = frames.pop_front().unwrap();

        // Parse handler ID
        if handler_id_bytes.len() != 4 {
            eprintln!(
                "[Dummy Backend] Invalid handler_id_bytes length: {}",
                handler_id_bytes.len()
            );
            continue;
        }

        let handler_id_array: [u8; 4] = handler_id_bytes
            .as_ref()
            .try_into()
            .context("Failed to convert handler_id_bytes to array")?;
        let handler_id = u32::from_be_bytes(handler_id_array);

        // Prepare response based on handler type
        let response_data = if handler_id == HANDSHAKE_ID {
            // Handle handshake request
            if frames.is_empty() {
                eprintln!("[Dummy Backend] Missing handshake request payload");
                continue;
            }
            let request_frame = frames.pop_front().unwrap();
            match rmp_serde::from_slice::<HandshakeRequest>(&request_frame) {
                Ok(_req) => {
                    // Return a minimal handshake response
                    let response = HandshakeResponse {
                        version: "0.1.0".to_string(),
                        model_name: "dummy-model".to_string(),
                        model_traits: vec![],
                        model_description: "Dummy backend for testing".to_string(),
                        prompt_template: "".to_string(),
                        prompt_template_type: "".to_string(),
                        prompt_stop_tokens: vec![],
                        kv_page_size: 16,
                        max_batch_tokens: 1024,
                        resources: HashMap::new(),
                        tokenizer_num_vocab: 0,
                        tokenizer_merge_table: HashMap::new(),
                        tokenizer_special_tokens: HashMap::new(),
                        tokenizer_split_regex: "".to_string(),
                        tokenizer_escape_non_printable: false,
                    };
                    Bytes::from(rmp_serde::to_vec_named(&response).unwrap())
                }
                Err(e) => {
                    eprintln!("[Dummy Backend] Failed to decode handshake request: {}", e);
                    continue;
                }
            }
        } else if handler_id == HEARTBEAT_ID {
            // Respond to heartbeat with an empty success response
            Bytes::from(rmp_serde::to_vec(&serde_json::json!({})).unwrap())
        } else {
            // For all other handlers, respond with an error
            let error_msg = format!(
                "Dummy backend does not implement handler ID: {}",
                handler_id
            );
            eprintln!("[Dummy Backend] {}", error_msg);
            Bytes::from(
                rmp_serde::to_vec(&serde_json::json!({
                    "error": error_msg
                }))
                .unwrap(),
            )
        };

        // Build response message: [client_identity, corr_id_bytes, handler_id_bytes, response_data]
        let mut response_frames: VecDeque<Bytes> = VecDeque::new();
        response_frames.push_back(client_identity);
        response_frames.push_back(corr_id_bytes);
        response_frames.push_back(handler_id_bytes);
        response_frames.push_back(response_data);

        let response_msg = ZmqMessage::try_from(response_frames)
            .map_err(|e| anyhow::anyhow!("Failed to create ZMQ response message: {:?}", e))?;

        socket
            .send(response_msg)
            .await
            .context("Failed to send ZMQ response")?;
    }
}
