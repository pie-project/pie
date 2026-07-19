use serde::{Deserialize, Serialize};

pub const PIE_DEVICE_PORT_EMBED_TOKENS: u32 = 1 << 0;
pub const PIE_DEVICE_PORT_PAGES: u32 = 1 << 1;
pub const PIE_DEVICE_PORT_POSITIONS: u32 = 1 << 2;
pub const PIE_DEVICE_PORT_PAGE_INDPTR: u32 = 1 << 3;
pub const PIE_DEVICE_PORT_W_SLOT: u32 = 1 << 4;
pub const PIE_DEVICE_PORT_KV_LEN: u32 = 1 << 5;
pub const PIE_DEVICE_PORT_W_OFF: u32 = 1 << 6;
/// The driver resolves a dense `AttnMask` descriptor channel (device-carried
/// bool mask cells) pre-forward. Orthogonal to the geometry-class port sets:
/// a masked device-resolved pass additionally requires this bit.
pub const PIE_DEVICE_PORT_ATTN_MASK: u32 = 1 << 7;

/// Full device-resolved port mask of the `DecodeEnvelope` geometry class.
/// Driver-neutral: any backend that executes the class resolves exactly these
/// ports on device.
pub const PIE_DECODE_ENVELOPE_PORTS: u32 =
    PIE_DEVICE_PORT_EMBED_TOKENS | PIE_DEVICE_PORT_POSITIONS | PIE_DEVICE_PORT_KV_LEN;

/// Full device-resolved port mask of the `DeviceGeometry` geometry class:
/// the program traces its complete explicit geometry in-graph and the driver
/// resolves every descriptor port from device channel cells pre-forward.
pub const PIE_DEVICE_GEOMETRY_PORTS: u32 = PIE_DEVICE_PORT_EMBED_TOKENS
    | PIE_DEVICE_PORT_PAGES
    | PIE_DEVICE_PORT_POSITIONS
    | PIE_DEVICE_PORT_PAGE_INDPTR
    | PIE_DEVICE_PORT_W_SLOT
    | PIE_DEVICE_PORT_KV_LEN
    | PIE_DEVICE_PORT_W_OFF;

/// Runtime-derived execution-geometry contract acknowledged at instance bind.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeometryClass {
    /// Geometry and values are supplied through the ordinary host wire.
    #[default]
    Host = 0,
    /// Decode slots come from declared shapes; token/length values are device-carried.
    DecodeEnvelope = 1,
    /// The program traces its complete explicit geometry in-graph (loop-carried
    /// pages/write descriptors); the driver resolves descriptor ports from
    /// device cells and the runtime only leases physical pages.
    DeviceGeometry = 2,
}

impl TryFrom<u32> for GeometryClass {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Host),
            1 => Ok(Self::DecodeEnvelope),
            2 => Ok(Self::DeviceGeometry),
            value => Err(value),
        }
    }
}
