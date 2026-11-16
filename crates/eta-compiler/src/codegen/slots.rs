use alloc::string::{String, ToString};

use eta_ir::op::tags;

use crate::codegen::op_view::OpView;

pub struct Slots {
    pub a0: String,
    pub a1: String,
    pub a2: String,
    pub o0: String,
    pub o1: String,
}

impl Slots {
    /// `blank` names the pointer an absent operand takes: any the kernel
    /// holds, never read.
    pub fn of(
        op: &OpView,
        base: u32,
        blank: &str,
        mut pointer: impl FnMut(u32) -> String,
    ) -> Self {
        let mut slots = Self {
            a0: blank.to_string(),
            a1: blank.to_string(),
            a2: blank.to_string(),
            o0: blank.to_string(),
            o1: blank.to_string(),
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
        if op.tag == tags::PIVOT_THRESHOLD {
            slots.a1 = pointer(op.pred_payload);
        }
        if op.results > 0 {
            slots.o0 = pointer(base);
        }
        if op.results > 1 {
            slots.o1 = pointer(base + 1);
        }
        slots
    }
}
