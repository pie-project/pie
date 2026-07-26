//! Which operand goes in which kernel slot.
//!
//! `ptir_m1_execute` takes five pointers -- three in, two out -- and the rule
//! for filling them from an op is part of the M1 op ABI, not part of any one
//! backend. It was written three times: `metal::fused::slots_for`, the same
//! function reused by the grouped path, and an inline copy in `cuda::fused`.
//!
//! The rules are small and every one of them is silent when wrong. If a
//! backend forgets that `pivot_threshold` passes its predicate payload in `a1`
//! it reads whatever the second operand slot held; if it forgets the second
//! result it writes one output and leaves the other pointing at scratch. Both
//! produce a kernel that compiles and runs.
//!
//! What is genuinely per-backend is how a value id becomes a pointer
//! expression -- Metal indexes a shared `offsets` table, CUDA resolves reshape
//! aliases first -- so that is the parameter.

use alloc::string::{String, ToString};

use pie_ir::op::tags;

use crate::op_view::OpView;

/// The five pointer slots, defaulted to `scratch` the way the C++ does before
/// the per-tag overrides.
pub struct Slots {
    pub a0: String,
    pub a1: String,
    pub a2: String,
    pub o0: String,
    pub o1: String,
}

impl Slots {
    /// Fill the slots for `op`, whose first result is value `base`.
    ///
    /// Slots the op does not use keep pointing at `scratch`; the runtime
    /// ignores them, and the emitters rely on that to leave a hole they then
    /// overwrite with a channel cell or an intrinsic base.
    pub fn of(op: &OpView, base: u32, mut pointer: impl FnMut(u32) -> String) -> Self {
        let mut slots = Self {
            a0: "scratch".to_string(),
            a1: "scratch".to_string(),
            a2: "scratch".to_string(),
            o0: "scratch".to_string(),
            o1: "scratch".to_string(),
        };
        if !op.args.is_empty() {
            slots.a0 = pointer(op.args[0]);
        }
        if op.args.len() > 1 {
            slots.a1 = pointer(op.args[1]);
        }
        if op.args.len() > 2 {
            slots.a2 = pointer(op.args[2]);
        }
        // `pivot_threshold` carries its threshold as a predicate payload
        // rather than an operand, and the runtime expects it in `a1`.
        if op.tag == tags::PIVOT_THRESHOLD {
            slots.a1 = pointer(op.pred_payload);
        }
        if op.results > 0 {
            slots.o0 = pointer(base);
        }
        // Results are consecutive value ids, so the second one is `base + 1`.
        if op.results > 1 {
            slots.o1 = pointer(base + 1);
        }
        slots
    }
}
