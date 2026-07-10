//! PTIR (pie tensor IR) engine surface — relocated out of `api/` (mechanical
//! move; the WIT host glue in `ptir_host` stays, the rest is engine domain
//! logic: registry, instance, channel store, geometry, KV/RS projection).
//! Names kept verbatim during the move; a later refactor renames
//! `ptir_host`→`host`, etc.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtirChannelValue {
    pub channel: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtirProgramSubmission {
    pub hash: u64,
    pub instance: u64,
    pub bytes: Option<Vec<u8>>,
    pub sidecar: Option<Vec<u8>>,
    pub channel_ids: Vec<u64>,
    pub seeds: Vec<PtirChannelValue>,
    pub host_puts: Vec<PtirChannelValue>,
}

pub mod ptir_channel_store;
pub mod ptir_geometry;
pub mod ptir_host;
pub mod ptir_instance;
pub mod ptir_kv;
pub mod ptir_lease;
pub mod ptir_registry;
pub mod ptir_rs;
