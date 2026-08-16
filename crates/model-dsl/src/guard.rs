//! THE RUNTIME BRANCH — `guarded(t).arm(pred, f).otherwise(f)`, the one
//! construct whose regions are chosen at fire time rather than traced away.

use super::*;

/// An open [`OpKind::Guard`](model_ir::trace::OpKind::Guard) chain: `.arm(pred, f)` regions are tried in
/// order at fire time, `.otherwise(f)` closes the chain with the
/// fallback region. The ONE branch a lowered declaration may write over
/// runtime inputs — the predicate vocabulary is closed ([`GuardPred`](model_ir::trace::GuardPred)),
/// the regions are flat and consecutive, and a region's OWN values may
/// not escape (its launches are lowerings of the guard's outputs, which
/// [`guarded_value`] created up front; the discipline is reviewed, not
/// enforced).
#[must_use = "a guard chain must be closed with .otherwise(..)"]
pub struct GuardCtx {
    // `pub(crate)` and not private: `rows.rs` builds one directly
    // (`rows!(..)` opens a guard whose single arm is the row predicate), and
    // that was a same-module construction until the surface became one file
    // per kind of statement. Crate-visible rather than public, so the fields
    // are reachable from the file that needs them and from nowhere a
    // declaration can write.
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

/// Open a side-effect-only guard chain.
///
/// Takes the TAPE, not a model context: a guard is a statement about what
/// gets recorded, and nothing about it is one family's. This used to have an
/// `&M`-taking twin whose whole body was `guarded(&m.t)`.
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

/// Open a VALUE-PRODUCING guard chain: the returned [`Val`]s are the
/// guard's outputs — one producer whichever arm runs — and each region's
/// launches are their lowerings, binding the same output buffer and
/// recording no SSA outputs of their own.
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

/// Two-way sugar over [`GuardCtx`] — the 4a form llama_like writes.
pub fn guard(
    t: &Trace,
    pred: model_ir::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
}
