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

#[derive(Debug, Serialize, Deserialize)]
pub struct InstanceInfo {
    pub id: String,
    pub cmd_name: String,
    pub arguments: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum StreamingOutput {
    Stdout(String),
    Stderr(String),
}

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "identification")]
    Identification { corr_id: u32, username: String },

    #[serde(rename = "signature")]
    Signature { corr_id: u32, signature: String },

    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: String,
        record: String,
    },

    #[serde(rename = "upload_program")]
    UploadProgram {
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "launch_instance")]
    LaunchInstance {
        corr_id: u32,
        program_hash: String,
        cmd_name: String,
        arguments: Vec<String>,
        detached: bool,
    },

    #[serde(rename = "launch_server_instance")]
    LaunchServerInstance {
        corr_id: u32,
        port: u32,
        program_hash: String,
        cmd_name: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "signal_instance")]
    SignalInstance {
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

    #[serde(rename = "terminate_instance")]
    TerminateInstance { instance_id: String },

    #[serde(rename = "attach_remote_service")]
    AttachRemoteService {
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    },

    #[serde(rename = "internal_authenticate")]
    InternalAuthenticate { corr_id: u32, token: String },

    #[serde(rename = "ping")]
    Ping { corr_id: u32 },

    #[serde(rename = "list_instances")]
    ListInstances { corr_id: u32 },
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

    #[serde(rename = "instance_event")]
    InstanceEvent {
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

    #[serde(rename = "live_instances")]
    LiveInstances {
        corr_id: u32,
        instances: Vec<InstanceInfo>,
    },

    #[serde(rename = "streaming_output")]
    StreamingOutput {
        instance_id: String,
        output: StreamingOutput,
    },
}
