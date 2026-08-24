use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardPlan {
    pub family: String,

    pub values: Vec<ValueInfo>,

    pub ops: Vec<Op>,

    #[serde(default, skip_serializing_if = "is_false")]
    pub depth_window: bool,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seams: Vec<SeamStatement>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime: Vec<RuntimeBinding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeBinding {
    pub name: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,

    pub value: ValueId,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ForwardPlan {
    // `depth_windowed` AND `depth_prefix_plan` STOOD HERE. The second asked
    // `KernelSig::depth_prefix_plan` — whether a launch inside a depth window
    // plans over the whole prefix — and the first existed to answer it. The
    // last reader of either was the legacy driver's depth walk, which went
    // with `model_compiler::lower` at R3; R4e measured zero callers and the
    // column they read went with them.
    pub fn layer_ops(&self, l: u32) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(move |op| op.layer == Some(l))
    }
}
