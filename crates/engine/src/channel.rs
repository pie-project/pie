use serde::{Deserialize, Serialize};

use eta_ir::container::{ChanDType, ExternDir, HostRole};

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct ChannelId(pub u64);

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelSeed {
    pub channel: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRegistration {
    pub id: ChannelId,
    pub shape: Vec<u32>,
    pub dtype: ChanDType,
    pub host_role: HostRole,
    pub seeded: bool,
    pub extern_dir: Option<ExternDir>,
    pub capacity: u32,
    pub extern_name: Vec<u8>,
}

impl Default for ChannelRegistration {
    fn default() -> ChannelRegistration {
        ChannelRegistration {
            id: ChannelId(0),
            shape: Vec::new(),
            dtype: ChanDType::Concrete(eta_ir::types::Dtype::F32),
            host_role: HostRole::None,
            seeded: false,
            extern_dir: None,
            capacity: 0,
            extern_name: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMirror {
    pub mirror: u64,
    pub words: u64,
    pub cell_bytes: u32,
    pub capacity: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredChannel {
    pub id: ChannelId,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    #[serde(skip)]
    pub mirror: Option<HostMirror>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ticket {
    pub channel: ChannelId,
    pub expected_head: u64,
    pub expected_tail: u64,
}

impl Ticket {
    pub const NONE: u64 = u64::MAX;
}
