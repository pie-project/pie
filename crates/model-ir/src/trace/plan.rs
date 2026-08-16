//! A traced forward pass as one serializable value.

use super::*;
use serde::{Deserialize, Serialize};

/// Traced form of one family's forward pass for one set of load-time facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardPlan {
    /// Family plus facts digest; used as a cache key and in reports.
    pub family: String,
    /// Values indexed by `ValueId`.
    pub values: Vec<ValueInfo>,
    /// The statements, in text order.
    pub ops: Vec<Op>,
    /// Enables per-layer depth windows via each op's `layer` tag.
    #[serde(default, skip_serializing_if = "is_false")]
    pub depth_window: bool,
    /// Every seam the text stated ([`SeamStatement`]), in text order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seams: Vec<SeamStatement>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ForwardPlan {
    /// Layer-tagged statements are skipped or windowed on depth-split fires.
    pub fn depth_windowed(&self, op: &Op) -> bool {
        self.depth_window && op.layer.is_some()
    }

    /// Whether this op's kernel uses the depth-prefix plan and workspace
    /// on union tail layers.
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
