//! Versioned worker-to-executor driver protocol.

use serde::{Deserialize, Serialize};

use crate::{
    DriverCapabilities, GeometryClass, KvCopyPlan, KvHandle, KvLayout, LaunchPlan,
    ProgramRegistration,
};

// v8: LaunchPlan carries the translated physical KV commit high-water.
pub const REMOTE_WIRE_VERSION: u32 = 8;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelComponent {
    Full,
    Text,
    Encode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub hash: [u8; 32],
    pub component: ModelComponent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RemoteTransferKind {
    Inline,
    Local,
    Nixl,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePeerConn {
    pub kind: RemoteTransferKind,
    pub handle: Option<KvHandle>,
    pub metadata: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScratchGrant {
    pub base_page: u32,
    pub num_pages: u32,
}

impl ScratchGrant {
    pub fn end_page(self) -> Option<u32> {
        self.base_page.checked_add(self.num_pages)
    }

    pub fn contains(self, page: u32) -> bool {
        self.end_page()
            .is_some_and(|end| page >= self.base_page && page < end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloRequest {
    pub wire_version: u32,
    pub client_nonce: u64,
    pub model: ModelIdentity,
    pub kv_layout: KvLayout,
    pub peer_conn: Option<RemotePeerConn>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelloResponse {
    pub wire_version: u32,
    pub model: ModelIdentity,
    pub kv_layout: KvLayout,
    pub capabilities: DriverCapabilities,
    pub grant: ScratchGrant,
    pub peer_conn: RemotePeerConn,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteChannelValue {
    pub channel_id: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteRegisterChannel {
    pub local_channel_id: u64,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub host_role: u8,
    pub seeded: bool,
    pub extern_dir: u8,
    pub capacity: u32,
    pub extern_name: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteChannelBinding {
    pub local_channel_id: u64,
    pub executor_channel_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBindInstance {
    pub local_instance_id: u64,
    pub program_id: u64,
    pub channel_ids: Vec<u64>,
    pub seed_values: Vec<RemoteChannelValue>,
    pub geometry_class: GeometryClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteBindResponse {
    pub local_instance_id: u64,
    pub executor_instance_id: u64,
    pub geometry_class: GeometryClass,
}

/// Process-independent portion of a local `LaunchSubmission`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteLaunch {
    pub plan: LaunchPlan,
    pub instance_ids: Vec<u64>,
    pub terminal_count: u32,
    pub kv_translation: Vec<u32>,
    pub kv_translation_indptr: Vec<u32>,
    pub program_row_indptr: Vec<u32>,
    pub logical_fire_ids: Vec<u64>,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
    pub channel_ticket_indptr: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalCellState {
    pub outcome: u32,
    pub reserved0: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteTerminal {
    pub per_request: Vec<TerminalCellState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushKv {
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub dst_worker: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineKvPayload {
    pub dst_page_ids: Vec<u32>,
    pub page_bytes: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteEncode {
    pub plan: LaunchPlan,
    pub blobs: Vec<RemoteMediaBlob>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteMediaKind {
    ImagePixels,
    AudioFeatures,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteMediaBlob {
    pub kind: RemoteMediaKind,
    pub hash: [u8; 32],
    pub size: u64,
    pub origin: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteEmbeddings {
    pub rows: Vec<u8>,
    pub indptr: Vec<u32>,
    pub shapes: Vec<u32>,
    pub dtypes: Vec<u8>,
    pub anchor_rows: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorRequest {
    Hello(HelloRequest),
    LoadedModel,
    RegisterProgram(ProgramRegistration),
    RegisterChannel(RemoteRegisterChannel),
    BindInstance(RemoteBindInstance),
    Launch(RemoteLaunch),
    CopyKv(KvCopyPlan),
    Encode(RemoteEncode),
    PushKv(PushKv),
    CloseInstance(u64),
    CloseChannel(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorResponse {
    Hello(HelloResponse),
    LoadedModel(bool),
    ProgramRegistered(u64),
    ChannelRegistered(RemoteChannelBinding),
    InstanceBound(RemoteBindResponse),
    Terminal(RemoteTerminal),
    KvPayload(InlineKvPayload),
    KvPushed,
    Embeddings(RemoteEmbeddings),
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteErrorKind {
    InvalidRequest,
    Incompatible,
    ResourceExhausted,
    Unsupported,
    Disconnected,
    Driver,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteError {
    pub kind: RemoteErrorKind,
    pub message: String,
}

impl RemoteError {
    pub fn new(kind: RemoteErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for RemoteError {}

#[tarpc::service]
pub trait ExecutorRpc {
    async fn execute(request: ExecutorRequest) -> Result<ExecutorResponse, RemoteError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncodedMask, KvDtype, KvLayoutKind};

    fn layout() -> KvLayout {
        KvLayout {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            page_size: 16,
            dtype: KvDtype::Bf16,
            kind: KvLayoutKind::KvSeparate,
            storage_format: "test-bf16".to_string(),
            region_page_bytes: Vec::new(),
        }
    }

    #[test]
    fn request_envelope_round_trips() {
        let request = ExecutorRequest::Launch(RemoteLaunch {
            plan: LaunchPlan {
                token_ids: vec![1, 2],
                position_ids: vec![0, 1],
                masks: vec![EncodedMask::new(vec![0, 2], 2)],
                qo_indptr: vec![0, 2],
                ..LaunchPlan::default()
            },
            instance_ids: vec![7],
            terminal_count: 1,
            kv_translation: vec![11],
            kv_translation_indptr: vec![0, 1],
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![9],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0, 0],
        });
        let bytes = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            serde_json::from_slice::<ExecutorRequest>(&bytes).unwrap(),
            request
        );
    }

    #[test]
    fn encode_blob_envelope_round_trips() {
        let request = ExecutorRequest::Encode(RemoteEncode {
            plan: LaunchPlan {
                image_pixel_indptr: vec![0, 16],
                image_anchor_rows: vec![3],
                ..LaunchPlan::default()
            },
            blobs: vec![RemoteMediaBlob {
                kind: RemoteMediaKind::ImagePixels,
                hash: [0x5a; 32],
                size: 16,
                origin: "http://127.0.0.1:9000".to_string(),
            }],
        });
        let bytes = serde_json::to_vec(&request).unwrap();
        assert_eq!(
            serde_json::from_slice::<ExecutorRequest>(&bytes).unwrap(),
            request
        );
    }

    #[test]
    fn hello_carries_exact_wire_and_layout_identity() {
        let request = HelloRequest {
            wire_version: REMOTE_WIRE_VERSION,
            client_nonce: 42,
            model: ModelIdentity {
                hash: [3; 32],
                component: ModelComponent::Full,
            },
            kv_layout: layout(),
            peer_conn: None,
        };
        assert_eq!(request.wire_version, REMOTE_WIRE_VERSION);
        assert!(request.kv_layout.compatible_with(&layout()));
        let mut incompatible = layout();
        incompatible.storage_format = "test-fp8-hnd".to_string();
        assert!(!request.kv_layout.compatible_with(&incompatible));
    }

    #[test]
    fn scratch_grant_bounds_are_checked() {
        let grant = ScratchGrant {
            base_page: 10,
            num_pages: 3,
        };
        assert!(!grant.contains(9));
        assert!(grant.contains(10));
        assert!(grant.contains(12));
        assert!(!grant.contains(13));
        assert_eq!(
            ScratchGrant {
                base_page: u32::MAX,
                num_pages: 2,
            }
            .end_page(),
            None
        );
    }
}
