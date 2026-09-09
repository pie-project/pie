use alloc::string::String;
use alloc::vec::Vec;

use super::op::IntrinsicId;
use crate::types::Dtype;

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
        Pages = 3, "pages";
        PageIndptr = 4, "page_indptr";
        KvLen = 5, "kv_len";
        WSlot = 6, "w_slot";
        WOff = 7, "w_off";
        Readout = 8, "readout";
        AttnMask = 9, "attn_mask";
        RsBufferPages = 10, "rs_buffer_pages";
        RsBufferIndptr = 11, "rs_buffer_indptr";
        RsBufferLen = 12, "rs_buffer_len";
        RsWSlot = 13, "rs_w_slot";
        RsWOff = 14, "rs_w_off";
        RsFoldLen = 15, "rs_fold_len";
    }
}

impl Port {
    pub fn consumes(self) -> bool {
        matches!(
            self,
            Port::EmbedTokens
                | Port::Positions
                | Port::WSlot
                | Port::WOff
                | Port::RsWSlot
                | Port::RsWOff
                | Port::RsFoldLen
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PortMask(u32);

impl PortMask {
    pub const NONE: PortMask = PortMask(0);

    pub const DECODE_ENVELOPE: PortMask =
        PortMask::of(&[Port::EmbedTokens, Port::Positions, Port::KvLen]);

    pub const DEVICE_GEOMETRY: PortMask = PortMask::of(&[
        Port::EmbedTokens,
        Port::Positions,
        Port::KvLen,
        Port::Pages,
        Port::PageIndptr,
        Port::WSlot,
        Port::WOff,
    ]);

    pub const RS_BUFFER: PortMask = PortMask::of(&[
        Port::RsBufferPages,
        Port::RsBufferIndptr,
        Port::RsBufferLen,
        Port::RsWSlot,
        Port::RsWOff,
    ]);

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

    #[must_use]
    pub fn admitted_by(served: PortMask) -> GeometryClass {
        if served.covers(PortMask::DEVICE_GEOMETRY) {
            GeometryClass::DeviceGeometry
        } else if served.covers(PortMask::DECODE_ENVELOPE) {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
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

pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits
        | IntrinsicId::MtpLogits
        | IntrinsicId::Hidden
        | IntrinsicId::Velocity
        | IntrinsicId::PeerVelocity
        | IntrinsicId::Pixels
        | IntrinsicId::ValueHead => &[Stage::Epilogue],
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        IntrinsicId::AttnScore => &[Stage::Epilogue],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

pub fn intrinsic_available(intr: IntrinsicId, profile: &ModelProfile) -> bool {
    match intr {
        IntrinsicId::MtpLogits => profile.has_mtp_logits,
        IntrinsicId::MtpDrafts => profile.mtp_depth > 0,
        IntrinsicId::ValueHead => profile.has_value_head,
        IntrinsicId::AttnScore => profile.has_attn_score,
        IntrinsicId::Velocity | IntrinsicId::PeerVelocity => profile.has_velocity,
        IntrinsicId::Pixels => profile.has_pixels,
        IntrinsicId::Logits | IntrinsicId::Hidden | IntrinsicId::Query | IntrinsicId::Layer => true,
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelInfo {
    pub name: String,
    pub sink_scope: Option<SinkScope>,
    pub replayable: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelProfile {
    pub vocab: u32,
    pub page_size: u32,
    pub num_layers: u32,
    pub activation: Dtype,
    pub has_mtp_logits: bool,
    pub mtp_depth: u32,
    pub draft_block: u32,
    pub draft_mask_token: u32,
    pub draft_bidirectional: bool,
    pub draft_proposals_from: u32,
    pub has_value_head: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_velocity: bool,
    pub velocity_width: u32,
    pub has_pixels: bool,
    pub pixels_width: u32,
    pub has_lora: bool,
    pub kernels: Vec<KernelInfo>,
}

impl ModelProfile {
    pub fn activation_name(&self) -> &'static str {
        crate::types::name_or_unknown(self.activation)
    }

    pub fn kernel(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.iter().find(|k| k.name == name)
    }

    pub fn dummy() -> Self {
        ModelProfile {
            vocab: 32,
            page_size: 4,
            num_layers: 2,
            activation: Dtype::F32,
            has_mtp_logits: true,
            mtp_depth: 1,
            draft_block: 0,
            draft_mask_token: 0,
            draft_bidirectional: false,
            draft_proposals_from: 1,
            has_value_head: true,
            has_attn_score: true,
            has_velocity: true,
            velocity_width: 8,
            has_pixels: true,
            pixels_width: 3,
            has_attn_page_mask: true,
            has_lora: true,
            kernels: Vec::new(),
        }
    }
}
