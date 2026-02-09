use serde::{Deserialize, Serialize};

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024; // 256 KiB
pub const QUERY_PROGRAM_EXISTS: &str = "program_exists";
pub const QUERY_MODEL_STATUS: &str = "model_status";
pub const QUERY_BACKEND_STATS: &str = "backend_stats";

#[derive(Debug, Serialize, Deserialize)]
pub enum EventCode {
    Message = 0,
    Completed = 1,
    Aborted = 2,
    Exception = 3,
    ServerError = 4,
    OutOfResources = 5,
}

impl EventCode {
    pub fn from_u32(code: u32) -> Option<EventCode> {
        match code {
            0 => Some(EventCode::Message),
            1 => Some(EventCode::Completed),
            2 => Some(EventCode::Aborted),
            3 => Some(EventCode::Exception),
            4 => Some(EventCode::ServerError),
            5 => Some(EventCode::OutOfResources),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstanceStatus {
    Attached,
    Detached,
    Finished,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    pub arguments: Vec<String>,
    pub status: InstanceStatus,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub elapsed_secs: u64,
    #[serde(default)]
    pub kv_pages_used: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingOutput {
    Stdout(String),
    Stderr(String),
}

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "auth_request")]
    AuthRequest { corr_id: u32, username: String },

    #[serde(rename = "auth_response")]
    AuthResponse { corr_id: u32, signature: String },

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
        arguments: Vec<String>,
        capture_outputs: bool,
    },

    #[serde(rename = "attach_process")]
    AttachProcess { corr_id: u32, instance_id: String },

    #[serde(rename = "launch_daemon")]
    LaunchDaemon {
        corr_id: u32,
        port: u32,
        inferlet: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "signal_process")]
    SignalProcess {
        instance_id: String,
        message: String,
    },

    #[serde(rename = "upload_blob")]
    UploadBlob {
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "terminate_process")]
    TerminateProcess { corr_id: u32, instance_id: String },

    #[serde(rename = "auth_by_token")]
    AuthByToken { corr_id: u32, token: String },

    #[serde(rename = "ping")]
    Ping { corr_id: u32 },

    #[serde(rename = "list_processes")]
    ListProcesses { corr_id: u32 },
}

/// Messages from server -> client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        successful: bool,
        result: String,
    },

    #[serde(rename = "process_launch_result")]
    ProcessLaunchResult {
        corr_id: u32,
        successful: bool,
        message: String,
    },

    #[serde(rename = "process_attach_result")]
    ProcessAttachResult {
        corr_id: u32,
        successful: bool,
        message: String,
    },

    #[serde(rename = "process_event")]
    ProcessEvent {
        instance_id: String,
        event: u32,
        message: String,
    },

    #[serde(rename = "download_blob")]
    DownloadBlob {
        corr_id: u32,
        instance_id: String,
        blob_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "server_event")]
    ServerEvent { message: String },

    #[serde(rename = "challenge")]
    Challenge { corr_id: u32, challenge: String },

    #[serde(rename = "live_processes")]
    LiveProcesses {
        corr_id: u32,
        instances: Vec<InstanceInfo>,
    },

    #[serde(rename = "streaming_output")]
    StreamingOutput {
        instance_id: String,
        output: StreamingOutput,
    },
}
