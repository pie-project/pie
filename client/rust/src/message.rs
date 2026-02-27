use serde::{Deserialize, Serialize};

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024; // 256 KiB
pub const QUERY_MODEL_STATUS: &str = "model_status";

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
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
        arguments: Vec<String>,
        capture_outputs: bool,
    },

    #[serde(rename = "launch_daemon")]
    LaunchDaemon {
        corr_id: u32,
        port: u32,
        inferlet: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "attach_process")]
    AttachProcess { corr_id: u32, process_id: String },

    #[serde(rename = "terminate_process")]
    TerminateProcess { corr_id: u32, process_id: String },

    #[serde(rename = "signal_process")]
    SignalProcess {
        process_id: String,
        message: String,
    },

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

    #[serde(rename = "register_mcp_server")]
    RegisterMcpServer {
        corr_id: u32,
        name: String,
        transport: String,
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        args: Option<Vec<String>>,
        #[serde(default)]
        url: Option<String>,
    },

    #[serde(rename = "mcp_response")]
    McpResponse {
        corr_id: u32,
        ok: bool,
        result: String,
    },

    #[serde(rename = "submit_workflow")]
    SubmitWorkflow {
        corr_id: u32,
        json: String,
    },

    #[serde(rename = "cancel_workflow")]
    CancelWorkflow {
        corr_id: u32,
        workflow_id: String,
    },

    #[serde(rename = "attach_workflow")]
    AttachWorkflow { corr_id: u32, workflow_id: String },

    #[serde(rename = "detach_workflow")]
    DetachWorkflow { corr_id: u32, workflow_id: String },
}

/// Messages from server -> client
#[derive(Debug, Serialize, Deserialize)]
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

    #[serde(rename = "mcp_request")]
    McpRequest {
        corr_id: u32,
        process_id: String,
        server_name: String,
        method: String,
        params: String,
    },
}
