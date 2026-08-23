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
    pub fn depth_windowed(&self, op: &Op) -> bool {
        self.depth_window && op.layer.is_some()
    }

    pub fn depth_prefix_plan(&self, op: &Op) -> bool {
        if !self.depth_windowed(op) {
            return false;
        }
        let OpKind::Launch { kernel, .. } = &op.kind else {
            return false;
        };
        crate::kernels::Backend::of_family(&self.family)
            .and_then(|b| crate::kernels::stated_in(b, kernel))
            .is_some_and(|k| k.depth_prefix_plan)
    }

    pub fn layer_ops(&self, l: u32) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(move |op| op.layer == Some(l))
    }
}
