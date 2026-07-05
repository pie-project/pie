//! Client ↔ server message vocabulary — the edge wire payloads.
//!
//! These are the messages a client exchanges with the server over the
//! client-facing edge (WebSocket today; gateway/worker tarpc in disaggregated
//! serving). The [`edge`](crate::edge) frames in this crate embed them, and the
//! public `pie-client` crate re-exports them.
//!
//! Plain serde vocabulary — deliberately NOT `#[schema]`/rkyv (it never rides
//! the zero-copy tensor ring), so it is NOT part of `SCHEMA_HASH`.

use serde::{Deserialize, Serialize};

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024; // 256 KiB
pub const QUERY_MODEL_STATUS: &str = "model_status";

/// Messages from client -> server
//
// `Clone` so the gateway's `Request` (which carries one) is cloneable for
// idempotent re-dispatch / retry across worker candidates (design §8).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "auth_identify")]
    AuthIdentify { corr_id: u32, username: String },

    #[serde(rename = "auth_prove")]
    AuthProve { corr_id: u32, signature: String },

    #[serde(rename = "auth_by_token")]
    AuthByToken { corr_id: u32, token: String },

    #[serde(rename = "check_program")]
    CheckProgram {
        corr_id: u32,
        name: String,
        version: String,
        #[serde(default)]
        wasm_hash: Option<String>,
        #[serde(default)]
        manifest_hash: Option<String>,
    },

    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: String,
        record: String,
    },

    #[serde(rename = "add_program")]
    AddProgram {
        corr_id: u32,
        program_hash: String,
        manifest: String,
        force_overwrite: bool,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "launch_process")]
    LaunchProcess {
        corr_id: u32,
        inferlet: String,
        input: String,
        capture_outputs: bool,
    },

    #[serde(rename = "attach_process")]
    AttachProcess { corr_id: u32, process_id: String },

    #[serde(rename = "terminate_process")]
    TerminateProcess { corr_id: u32, process_id: String },

    #[serde(rename = "signal_process")]
    SignalProcess { process_id: String, message: String },

    #[serde(rename = "transfer_file")]
    TransferFile {
        process_id: String,
        file_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "list_processes")]
    ListProcesses { corr_id: u32 },

    #[serde(rename = "ping")]
    Ping { corr_id: u32 },
}

/// Messages from server -> client
//
// `Clone` so the gateway's `Tokens` chunk (which carries these) is cloneable on
// the streaming/fan-out path.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        ok: bool,
        result: String,
    },

    #[serde(rename = "process_event")]
    ProcessEvent {
        process_id: String,
        event: String,
        value: String,
    },

    #[serde(rename = "file")]
    File {
        process_id: String,
        file_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },
}
