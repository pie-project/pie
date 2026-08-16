//! Row/fire region constructs for traced partitions.

use super::*;

/// Row predicates partition a fire; unlike guards, both regions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPred {
    HookFree,
    Unmasked,
}

impl RowPred {
    /// Axis word derived from predicate, not restated by callers.
    fn window(self) -> model_ir::trace::PeelWindow {
        match self {
            RowPred::HookFree => model_ir::trace::PeelWindow::HookFreePrefix,
            RowPred::Unmasked => model_ir::trace::PeelWindow::UnmaskedPrefix,
        }
    }
}

/// Open row partition; `.rest(..)` must close it.
#[must_use = "a row partition must be closed with .rest(..)"]
pub struct RowsCtx<'t> {
    t: &'t Trace,
    idx: usize,
    prefix: Option<u32>,
    pred: Option<RowPred>,
}

impl RowsCtx<'_> {
    /// First region; today there is one arm plus rest.
    pub fn arm(&mut self, pred: RowPred, f: impl FnOnce()) {
        assert!(
            self.pred.is_none(),
            "by_rows takes one arm and a rest today — the IR's Peel is a \
             two-region op (`.wiki/tart/dsl.md` migration step 6 flattens it)"
        );
        f();
        let b = self.t.inner.borrow();
        self.prefix = Some((b.op_count_now() - self.idx - 1) as u32);
        drop(b);
        self.pred = Some(pred);
    }

    pub fn rest(&mut self, f: impl FnOnce()) {
        let prefix = self
            .prefix
            .expect("a row partition states its arm before its rest");
        f();
        let mut b = self.t.inner.borrow_mut();
        let total = (b.op_count_now() - self.idx - 1) as u32;
        b.close_peel(self.idx, prefix, total - prefix);
        b.set_peel_window(self.idx, self.pred.expect("an arm was stated").window());
    }
}

/// Region discipline: fire-exclusive guard or row partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Fire(model_ir::trace::GuardPred),
    Rows(RowPred),
}

/// One construct, but first arm fixes its discipline.
pub struct RegionsCtx<'t> {
    guard: Option<GuardCtx>,
    rows: Option<RowsCtx<'t>>,
    t: &'t Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    out: Option<Val>,
}

impl<'t> RegionsCtx<'t> {
    /// Mixed fire/row chains must be nested, not flattened.
    pub fn arm(&mut self, pred: Region, f: impl FnOnce()) {
        match pred {
            Region::Fire(p) => {
                assert!(
                    self.rows.is_none(),
                    "regions: a Fire arm after a Rows arm — one flat chain cannot be both disciplines (nest instead; the IR merge is migration step 6)"
                );
                let g = self.guard.take().unwrap_or_else(|| {
                    let (idx, outs) = {
                        let mut b = self.t.inner.borrow_mut();
                        b.set_layer(self.layer);
                        b.open_guard(self.shape.clone().into_iter().collect())
                    };
                    self.out = outs.first().map(|v| Val {
                        t: self.t.clone(),
                        id: *v,
                        layer: self.layer,
                    });
                    GuardCtx {
                        t: self.t.clone(),
                        idx,
                        arms: Vec::new(),
                        emitted: 0,
                    }
                });
                self.guard = Some(g.arm(p, f));
            }
            Region::Rows(p) => {
                assert!(
                    self.guard.is_none(),
                    "regions: a Rows arm after a Fire arm — one flat chain cannot be both disciplines (nest instead; the IR merge is migration step 6)"
                );
                let ctx = self.rows.get_or_insert_with(|| {
                    let (idx, outs) = {
                        let mut b = self.t.inner.borrow_mut();
                        b.set_layer(self.layer);
                        // Axis patched at close after the arm names it.
                        b.open_peel(
                            self.shape.clone().into_iter().collect(),
                            model_ir::trace::PeelWindow::HookFreePrefix,
                        )
                    };
                    self.out = outs.first().map(|v| Val {
                        t: self.t.clone(),
                        id: *v,
                        layer: self.layer,
                    });
                    RowsCtx {
                        t: self.t,
                        idx,
                        prefix: None,
                        pred: None,
                    }
                });
                ctx.arm(p, f);
            }
        }
    }

    fn close(mut self, f: impl FnOnce()) -> Option<Val> {
        if let Some(g) = self.guard.take() {
            g.otherwise(f);
        } else if let Some(mut r) = self.rows.take() {
            r.rest(f);
        } else {
            panic!("regions states at least one arm before its rest");
        }
        self.out
    }
}

/// Common surface for guard/peel; lowers to the current IR op by arm kind.
pub fn regions(
    t: &Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    build: impl FnOnce(&mut RegionsCtx<'_>),
    rest: impl FnOnce(),
) -> Option<Val> {
    let mut ctx = RegionsCtx {
        guard: None,
        rows: None,
        t,
        layer,
        shape,
        out: None,
    };
    build(&mut ctx);
    ctx.close(rest)
}

/// Row partition; `shape` makes it value-producing over disjoint windows.
pub fn by_rows(
    t: &Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    build: impl FnOnce(&mut RowsCtx<'_>),
) -> Option<Val> {
    let (idx, outs) = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(layer);
        // Axis patched at close after the arm names it.
        b.open_peel(
            shape.into_iter().collect(),
            model_ir::trace::PeelWindow::HookFreePrefix,
        )
    };
    let mut ctx = RowsCtx {
        t,
        idx,
        prefix: None,
        pred: None,
    };
    build(&mut ctx);
    assert!(
        ctx.pred.is_some(),
        "a row partition must state an arm and a rest"
    );
    outs.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// Two-way guard over a bare `Trace`.
pub fn guard_on(
    t: &Trace,
    pred: model_ir::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
}

/// Record a hook site inside the hooked-fire guard arm.
pub fn hook_site(stage: model_ir::trace::HookStage, q: &Val, layer: u32) {
    q.t.with(Some(layer), |b| {
        b.push_hook_site(stage, layer, q.id);
    });
}
