//! THE PLAN — a traced forward pass, as one serializable value.

use super::*;
use serde::{Deserialize, Serialize};

/// The traced form of one family's forward pass, for one set of load-time
/// facts. Serializable so goldens can pin it and a driver can consume it
/// across the (future) C ABI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardPlan {
    /// The family that traced this, plus a facts digest — a cache key, and
    /// the first thing a mismatch report prints.
    pub family: String,
    pub values: Vec<ValueInfo>,
    pub ops: Vec<Op>,
    /// STRUCTURAL S-3: the DECLARATION states the depth axis — every
    /// layer-tagged op of this trace may run over the full-depth prefix
    /// row window when the fire plans a depth split (layers `[k, L)` at
    /// rows `[0, split)`), and may be SKIPPED entirely on a uniform
    /// truncated fire. The trace is layer-unrolled while `k` is a
    /// runtime input, so the axis is a trace-level capability keyed on
    /// each op's own `layer` tag, not a region op at a static position
    /// (the [`OpKind::Peel`] doc's row-window vocabulary, applied
    /// per-layer). False for classes whose bodies cannot window
    /// (XQA-deployment, padded head dims, prefill shapes).
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
    /// DEPTH HAS NO SYNTAX (`.wiki/tart/dsl.md` ③, migration step 5).
    ///
    /// Every statement tagged with a layer is implicitly `rows(depth >
    /// l)`: it is skipped on a uniform truncated fire once its layer
    /// reaches `k`, and runs over the full-depth prefix rows on a union
    /// fire. The author writes nothing, and the IR carries no word —
    /// membership is the LAYER TAG plus the declaration's axis, which
    /// is what an `Op` already has.
    ///
    /// This replaces a per-op `DepthRole` enum whose `Windowed` variant
    /// was exactly this predicate, restated on every layer-tagged op of
    /// every trace.
    pub fn depth_windowed(&self, op: &Op) -> bool {
        self.depth_window && op.layer.is_some()
    }

    /// Does this op's kernel pair the depth PREFIX plan (and its
    /// dedicated workspace) on union tail layers, instead of the fire's
    /// own decode plan?
    ///
    /// The other half of the retired `DepthRole`, and it was never a
    /// property of the OP: it is a property of the KERNEL, so it is asked
    /// of the backend beside `whole`
    /// ([`crate::kernels::Stated::depth_prefix_plan`]).
    ///
    /// Asked through [`crate::kernels::stated_in`] rather than a table.
    /// Metal has no rows left, so a table lookup answers `None` for every
    /// Metal symbol and this predicate would be false everywhere — the
    /// union's tail layers would take the fire's own decode plan and its
    /// workspace, silently, with nothing refusing.
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

    /// Ops belonging to layer `l`, in execution order.
    pub fn layer_ops(&self, l: u32) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(move |op| op.layer == Some(l))
    }
}
