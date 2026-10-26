use alloc::string::String;

use crate::types::Dtype;

crate::declare_tagged_enum! {
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum Stage {
        Prologue = 0, "prologue";
        Epilogue = 3, "epilogue";
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Prologue,
    Descriptor,
    Epilogue,
}

pub const PHASE_DESCRIPTOR_TAG: u8 = 0xFF;

impl Phase {
    pub fn tag(self) -> u8 {
        match self {
            Phase::Prologue => Stage::Prologue as u8,
            Phase::Descriptor => PHASE_DESCRIPTOR_TAG,
            Phase::Epilogue => Stage::Epilogue as u8,
        }
    }
    pub fn of_stage(s: Stage) -> Phase {
        match s {
            Stage::Prologue => Phase::Prologue,
            Stage::Epilogue => Phase::Epilogue,
        }
    }
    pub const ORDER: [Phase; 3] = [Phase::Prologue, Phase::Descriptor, Phase::Epilogue];
}

crate::declare_tagged_enum! {
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum Port {
        EmbedTokens = 0, "embed_tokens";
        EmbedIndptr = 1, "embed_indptr";
        Positions = 2, "positions";
        KvLen = 5, "kv_len";
        Readout = 8, "readout";
        AttnMask = 9, "attn_mask";
        Slots = 16, "slots";
        Fold = 17, "fold";
    }
}

impl Port {
    pub fn consumes(self) -> bool {
        matches!(self, Port::EmbedTokens | Port::Positions | Port::Slots)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PortMask(u32);

impl PortMask {
    pub const NONE: PortMask = PortMask(0);

    pub const DECODE_ENVELOPE: PortMask = PortMask::of(&[Port::EmbedTokens]);

    pub const DEVICE_GEOMETRY: PortMask =
        PortMask::of(&[Port::EmbedTokens, Port::Positions, Port::Slots, Port::KvLen]);

    pub const fn of(ports: &[Port]) -> PortMask {
        let mut bits = 0u32;
        let mut index = 0;
        while index < ports.len() {
            bits |= 1u32 << (ports[index] as u8);
            index += 1;
        }
        PortMask(bits)
    }

    #[must_use]
    pub const fn from_bits(bits: u32) -> PortMask {
        PortMask(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, port: Port) -> bool {
        self.0 & (1u32 << (port as u8)) != 0
    }

    #[must_use]
    pub const fn covers(self, other: PortMask) -> bool {
        self.0 & other.0 == other.0
    }

    #[must_use]
    pub const fn with(self, port: Port) -> PortMask {
        PortMask(self.0 | (1u32 << (port as u8)))
    }

    #[must_use]
    pub const fn union(self, other: PortMask) -> PortMask {
        PortMask(self.0 | other.0)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Port> {
        Port::ALL.iter().copied().filter(move |&p| self.contains(p))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GeometryClass {
    #[default]
    Host,
    DecodeEnvelope,
    DeviceGeometry,
}

impl GeometryClass {
    #[must_use]
    pub const fn ports(self) -> PortMask {
        match self {
            GeometryClass::Host => PortMask::NONE,
            GeometryClass::DecodeEnvelope => PortMask::DECODE_ENVELOPE,
            GeometryClass::DeviceGeometry => PortMask::DEVICE_GEOMETRY,
        }
    }
}

/// The sinks a prologue may call.
pub const KNOWN_SINKS: &[&str] = &["attn_page_mask", "lora", "minference_sparse"];

/// Whether an output is one row per lane, per row or per voxel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Axis {
    Lane,
    Row,
    Voxel,
}

/// An output of the model as a lane reads it: `width` values of `dtype` an
/// `axis` entry, stacked `layers` deep when written per layer (0 when not).
/// `readout`: its rows are the lane's readout rows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OutputDecl {
    pub name: String,
    pub axis: Axis,
    pub width: u32,
    pub dtype: Dtype,
    pub layers: u32,
    pub readout: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineCaps {
    pub lora: bool,
    pub attn_page_mask: bool,
}
