use super::op::IntrinsicId;

crate::declare_tagged_enum! {
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum Stage {
        Prologue = 0, "prologue";
        OnAttnProj = 1, "on_attn_proj";
        OnAttn = 2, "on_attn";
        Epilogue = 3, "epilogue";
    }
}

impl Stage {
    pub fn per_layer(self) -> bool {
        matches!(self, Stage::OnAttnProj | Stage::OnAttn)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Prologue,
    Descriptor,
    OnAttnProj,
    OnAttn,
    Epilogue,
}

pub const PHASE_DESCRIPTOR_TAG: u8 = 0xFF;

impl Phase {
    pub fn tag(self) -> u8 {
        match self {
            Phase::Prologue => Stage::Prologue as u8,
            Phase::Descriptor => PHASE_DESCRIPTOR_TAG,
            Phase::OnAttnProj => Stage::OnAttnProj as u8,
            Phase::OnAttn => Stage::OnAttn as u8,
            Phase::Epilogue => Stage::Epilogue as u8,
        }
    }
    pub fn of_stage(s: Stage) -> Phase {
        match s {
            Stage::Prologue => Phase::Prologue,
            Stage::OnAttnProj => Phase::OnAttnProj,
            Stage::OnAttn => Phase::OnAttn,
            Stage::Epilogue => Phase::Epilogue,
        }
    }
    pub const ORDER: [Phase; 5] = [
        Phase::Prologue,
        Phase::Descriptor,
        Phase::OnAttnProj,
        Phase::OnAttn,
        Phase::Epilogue,
    ];
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SinkScope {
    PassWide,
    Attention,
}

pub const KNOWN_SINKS: &[(&str, SinkScope)] = &[
    ("attn_page_mask", SinkScope::Attention),
    ("lora", SinkScope::PassWide),
    ("minference_sparse", SinkScope::PassWide),
];

pub const ATTN_SCORE_KV_MAX: u32 = 2048;

/// The last this many query rows of a fire's lane.
pub const ATTN_SCORE_OBSERVE: u32 = 32;

pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits
        | IntrinsicId::MtpLogits
        | IntrinsicId::Hidden
        | IntrinsicId::Velocity
        | IntrinsicId::Pixels => &[Stage::Epilogue],
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        IntrinsicId::AttnScore => &[Stage::Epilogue],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Readout {
    pub kind: ReadoutKind,
    pub width: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ReadoutKind {
    Logits,
    Velocity,
    Hidden,
    Pixels,
}

/// The model's graph facts a program may read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Graph {
    pub num_layers: u32,
    pub mtp_logits: bool,
    pub mtp_depth: u32,
    pub attn_score: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineCaps {
    pub lora: bool,
    pub attn_page_mask: bool,
}
