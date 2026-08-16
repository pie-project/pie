//! Runtime branch construct for guard regions.

use super::*;

/// Open guard chain; arms are tried in order and `.otherwise` closes it.
#[must_use = "a guard chain must be closed with .otherwise(..)"]
pub struct GuardCtx {
    // Crate-visible: `rows.rs` constructs this for unified regions.
    pub(crate) t: Trace,
    pub(crate) idx: usize,
    pub(crate) arms: Vec<model_ir::trace::GuardArm>,
    pub(crate) emitted: u32,
}

impl GuardCtx {
    pub fn arm(mut self, pred: model_ir::trace::GuardPred, f: impl FnOnce()) -> Self {
        f();
        let total = {
            let b = self.t.inner.borrow();
            (b.op_count_now() - self.idx - 1) as u32
        };
        self.arms.push(model_ir::trace::GuardArm {
            pred,
            ops: total - self.emitted,
        });
        self.emitted = total;
        self
    }

    pub fn otherwise(self, f: impl FnOnce()) {
        f();
        let mut b = self.t.inner.borrow_mut();
        let total = (b.op_count_now() - self.idx - 1) as u32;
        b.close_guard(self.idx, self.arms, total - self.emitted);
    }
}

/// Open a side-effect-only guard on the tape.
pub fn guarded(t: &Trace) -> GuardCtx {
    let idx = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(None);
        b.open_guard(vec![]).0
    };
    GuardCtx {
        t: t.clone(),
        idx,
        arms: Vec::new(),
        emitted: 0,
    }
}

/// Value-producing guard; region launches lower the guard output buffers.
pub fn guarded_value(t: &Trace, layer: Option<u32>, shape: (Shape, DType)) -> (GuardCtx, Val) {
    let (idx, outs) = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(layer);
        b.open_guard(vec![shape])
    };
    (
        GuardCtx {
            t: t.clone(),
            idx,
            arms: Vec::new(),
            emitted: 0,
        },
        Val {
            t: t.clone(),
            id: outs[0],
            layer,
        },
    )
}

/// Two-way sugar over `GuardCtx`.
pub fn guard(
    t: &Trace,
    pred: model_ir::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
}
