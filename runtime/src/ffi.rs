//! FFI module containing all Python-Rust interop infrastructure.
//!
//! This module serves as the boundary between Rust and Python, providing:
//! - `format`: Request/response types for IPC serialization
//! - `rpc`: IPC backend and client for cross-process communication
//! - `pybindings`: PyO3 bindings exposed to Python
//!
//! # Architecture
//!
//! The FFI layer uses ipc-channel for cross-process communication between
//! the Rust runtime (rank 0) and Python worker processes. This provides
//! GIL isolation in the symmetric worker architecture.

pub mod format;
pub mod pybindings;
pub mod rpc;

// Re-export commonly used types at the module level
pub use format::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ByteVec, ByteVecF32,
    DownloadAdapterRequest, EmbedImageRequest, ForwardPassRequest, ForwardPassResponse,
    HandshakeRequest, HandshakeResponse, InitializeAdapterRequest, QueryRequest, QueryResponse,
    Request, UpdateAdapterRequest, UploadAdapterRequest,
    DOWNLOAD_ADAPTER_ID, EMBED_IMAGE_ID, FORWARD_PASS_ID, HANDSHAKE_ID, INITIALIZE_ADAPTER_ID,
    QUERY_ID, UPDATE_ADAPTER_ID, UPLOAD_ADAPTER_ID,
};

pub use rpc::{AsyncIpcClient, FfiIpcBackend, IpcChannels, IpcRequest, IpcResponse, RpcBackend};

pub use pybindings::{
    _pie, FfiIpcQueue, PartialServerHandle, ServerConfig, ServerHandle,
};
