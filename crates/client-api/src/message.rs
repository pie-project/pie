use serde::{Deserialize, Serialize};

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024;
pub const QUERY_MODEL_STATUS: &str = "model_status";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "auth_identify")]
    AuthIdentify { corr_id: u32, username: String },

    #[serde(rename = "auth_prove")]
    AuthProve { corr_id: u32, signature: String },

    #[serde(rename = "check_program")]
    CheckProgram {
        corr_id: u32,
        name: String,
        version: String,
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
        file: String,
        version: Option<String>,
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

impl ClientMessage {
    pub fn corr_id(&self) -> Option<u32> {
        match self {
            ClientMessage::AuthIdentify { corr_id, .. }
            | ClientMessage::AuthProve { corr_id, .. }
            | ClientMessage::CheckProgram { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::AddProgram { corr_id, .. }
            | ClientMessage::LaunchProcess { corr_id, .. }
            | ClientMessage::AttachProcess { corr_id, .. }
            | ClientMessage::TerminateProcess { corr_id, .. }
            | ClientMessage::ListProcesses { corr_id }
            | ClientMessage::Ping { corr_id } => Some(*corr_id),
            ClientMessage::SignalProcess { .. } | ClientMessage::TransferFile { .. } => None,
        }
    }
}

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
        #[serde(default)]
        name: Option<String>,
    },
}
