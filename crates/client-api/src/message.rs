use serde::{Deserialize, Serialize};

use crate::Props;

/// Files a launch attaches, by name; the guest reads them with `session.file`.
pub type Files = std::collections::BTreeMap<String, serde_bytes::ByteBuf>;

/// A session's own name for a handle: the client picks it, as 9P picks a fid,
/// so a name can be used before the server has answered for it.
pub type HandleName = u64;

/// Handles a launch deals in, by the name the guest knows each under.
pub type Handles = std::collections::BTreeMap<String, HandleName>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandleKind {
    WorkingSet,
    Matcher(GrammarSource),
}

/// A grammar as a client writes it. It compiles where a process binds it, the
/// tokenizer being the worker's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSource {
    JsonSchema(String),
    Regex(String),
    Ebnf(String),
    Json,
}

/// What a client can do to a handle without running anything: its lifetime,
/// never its contents. Computing on one is a process's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum HandleOp {
    /// Names an empty handle. Nothing exists until a process binds it.
    New {
        name: HandleName,
        kind: HandleKind,
    },
    /// `name` starts as `from` stands. Neither sees the other's writes.
    Fork {
        name: HandleName,
        from: HandleName,
    },
    Close {
        name: HandleName,
    },
    Stat {
        name: HandleName,
    },
}

impl HandleOp {
    /// The names this op needs settled before it runs.
    pub fn touches(&self) -> impl Iterator<Item = HandleName> {
        let (first, second) = match self {
            HandleOp::New { name, .. } | HandleOp::Close { name } | HandleOp::Stat { name } => {
                (*name, None)
            }
            HandleOp::Fork { name, from } => (*name, Some(*from)),
        };
        std::iter::once(first).chain(second)
    }
}

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
        // Always written: msgpack reads fields by place, so none may be
        // skipped.
        #[serde(default)]
        props: Props,
        #[serde(default)]
        session: Option<String>,
        #[serde(default)]
        files: Files,
        /// Handles moved into the process, by the name it takes each under.
        #[serde(default)]
        handles: Handles,
        /// What the process gives back: the name each lands on in the session.
        #[serde(default)]
        returns: Handles,
        /// Set by the worker, never on the wire.
        #[serde(skip)]
        cluster_props: Props,
        /// The workers the cluster names to run its passes, best first, by
        /// id; set by the worker, never on the wire.
        #[serde(skip)]
        executors: Vec<String>,
    },

    /// Run in order; each op answers in the response's JSON array, and one
    /// that fails does not stop the rest.
    #[serde(rename = "handles")]
    Handles { corr_id: u32, ops: Vec<HandleOp> },

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
    /// Whether this frame deals in handles: such frames stay in order with
    /// each other, on the worker that holds the session's handles.
    #[must_use]
    pub fn uses_handles(&self) -> bool {
        match self {
            ClientMessage::Handles { .. } => true,
            ClientMessage::LaunchProcess {
                handles, returns, ..
            } => !handles.is_empty() || !returns.is_empty(),
            _ => false,
        }
    }

    pub fn corr_id(&self) -> Option<u32> {
        match self {
            ClientMessage::AuthIdentify { corr_id, .. }
            | ClientMessage::AuthProve { corr_id, .. }
            | ClientMessage::CheckProgram { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::AddProgram { corr_id, .. }
            | ClientMessage::LaunchProcess { corr_id, .. }
            | ClientMessage::Handles { corr_id, .. }
            | ClientMessage::AttachProcess { corr_id, .. }
            | ClientMessage::TerminateProcess { corr_id, .. }
            | ClientMessage::ListProcesses { corr_id }
            | ClientMessage::Ping { corr_id } => Some(*corr_id),
            ClientMessage::SignalProcess { .. } | ClientMessage::TransferFile { .. } => None,
        }
    }

    /// The request's correlation id to stamp, for the messages that carry one.
    pub fn corr_id_mut(&mut self) -> Option<&mut u32> {
        match self {
            ClientMessage::AuthIdentify { corr_id, .. }
            | ClientMessage::AuthProve { corr_id, .. }
            | ClientMessage::CheckProgram { corr_id, .. }
            | ClientMessage::Query { corr_id, .. }
            | ClientMessage::AddProgram { corr_id, .. }
            | ClientMessage::LaunchProcess { corr_id, .. }
            | ClientMessage::Handles { corr_id, .. }
            | ClientMessage::AttachProcess { corr_id, .. }
            | ClientMessage::TerminateProcess { corr_id, .. }
            | ClientMessage::ListProcesses { corr_id }
            | ClientMessage::Ping { corr_id } => Some(corr_id),
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
