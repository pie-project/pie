use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    #[doc(hidden)]
    Retired0,
    #[doc(hidden)]
    Retired1,
    #[doc(hidden)]
    Retired2,
    #[doc(hidden)]
    Retired3,
    #[doc(hidden)]
    Retired4,
    #[doc(hidden)]
    Retired5,
    #[doc(hidden)]
    Retired6,
    #[doc(hidden)]
    Retired7,
    #[doc(hidden)]
    Retired8,
    #[doc(hidden)]
    Retired9,

    LmHead {
        weight: String,
    },
    #[doc(hidden)]
    Retired11,

    Select {
        index: u32,
    },
    #[doc(hidden)]
    Retired13,
    #[doc(hidden)]
    Retired14,
    #[doc(hidden)]
    Retired15,
    #[doc(hidden)]
    Retired16,
    #[doc(hidden)]
    Retired17,
    #[doc(hidden)]
    Retired18,
    #[doc(hidden)]
    Retired19,
    #[doc(hidden)]
    Retired20,
    #[doc(hidden)]
    Retired21,
    #[doc(hidden)]
    Retired22,

    Launch {
        kernel: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        weights: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<StateRef>,

        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        params: Vec<u32>,

        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        param_extents: Vec<(u8, Shape)>,

        #[serde(default, skip_serializing_if = "Option::is_none")]
        peel_slots: Option<(u8, u8)>,
    },

    Guard {
        arms: Vec<GuardArm>,
        else_ops: u32,
    },

    HookSite {
        stage: HookStage,
        layer: u32,
    },

    Prep {
        prep: PrepKind,
    },

    Peel {
        prefix_ops: u32,
        tail_ops: u32,
        #[serde(default, skip_serializing_if = "PeelWindow::is_hook_free")]
        window: PeelWindow,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeelWindow {
    #[default]
    HookFreePrefix,

    UnmaskedPrefix,
}

impl PeelWindow {
    pub fn is_hook_free(&self) -> bool {
        matches!(self, PeelWindow::HookFreePrefix)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateStore {
    KvCache,

    RecurrentState,
}

impl StateStore {
    #[must_use]
    pub fn runtime_name(&self) -> &'static str {
        match self {
            StateStore::KvCache => "kv_cache",
            StateStore::RecurrentState => "recurrent_state",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRef {
    pub store: StateStore,

    pub layer: u32,
}

impl OpKind {
    pub fn state_ref(&self) -> Option<StateRef> {
        match *self {
            OpKind::Launch { state, .. } => state,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Op {
    pub kind: OpKind,

    pub inputs: Vec<ValueId>,

    pub outputs: Vec<ValueId>,

    pub layer: Option<u32>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dest: Vec<ValueId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrepKind {
    DecodeAttention { head_dim: u32, full_attention: bool },

    PrefillAttention { head_dim: u32 },
}

impl PrepKind {
    #[must_use]
    pub const fn key(self) -> &'static str {
        use kernels::raises::Raise;
        match self {
            Self::DecodeAttention { .. } => kernels_cuda::raises::Fa2Decode::KEY,
            Self::PrefillAttention { .. } => kernels_cuda::raises::Fa2Prefill::KEY,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    pub shape: Shape,

    pub dtype: DType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyn_axis: Option<DynAxis>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raised: Option<String>,
}

impl ValueInfo {
    #[must_use]
    pub fn raise(key: &str) -> Self {
        Self {
            shape: Shape(Vec::new()),
            dtype: DType::I32,
            dyn_axis: None,
            raised: Some(key.to_string()),
        }
    }

    #[must_use]
    pub fn is_raised(&self) -> bool {
        self.raised.is_some()
    }
}
