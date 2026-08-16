//! THE ROW PARTITION — `rows!(..)`: WHICH ROWS of a fire an arm's
//! statements cover, as a predicate the lowering turns into rectangles.

use super::*;

/// WHICH ROWS of the fire an arm's statements cover
/// (`.wiki/tart/dsl.md` ③'s `rows!(..)`).
///
/// A row predicate is not a deployment condition: it does not resolve at
/// trace time and vanish, it PARTITIONS the fire. Today's tree writes
/// both kinds as plain Rust `if`, which is why a reader cannot tell
/// which one disappears — naming the row kind is the first half of
/// fixing that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPred {
    /// Rows with nothing attached at a seam — the hook-free prefix.
    HookFree,
    /// Rows carrying no custom mask.
    Unmasked,
}

impl RowPred {
    /// The axis word today's IR carries. It is DERIVED here rather than
    /// passed: the arm's predicate already says which rows it covers, so
    /// stating the axis beside it was the same fact twice.
    fn window(self) -> model_ir::trace::PeelWindow {
        match self {
            RowPred::HookFree => model_ir::trace::PeelWindow::HookFreePrefix,
            RowPred::Unmasked => model_ir::trace::PeelWindow::UnmaskedPrefix,
        }
    }
}

/// The arms of a [`by_rows`] partition.
///
/// Each arm's statements record as the arm is written — the construct is
/// already open — and the axis word the IR carries is patched in at
/// close from the arm's predicate.
#[must_use = "a row partition must be closed with .rest(..)"]
pub struct RowsCtx<'t> {
    t: &'t Trace,
    idx: usize,
    prefix: Option<u32>,
    pred: Option<RowPred>,
}

impl RowsCtx<'_> {
    /// The rows `pred` names, and what runs over them.
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

    /// Every other row.
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

/// Which discipline an arm of [`regions`] follows.
///
/// The Guard/Peel unification is of the SURFACE, and this enum is where
/// that is said out loud: one construct in the text, arms that state
/// their own rule. Read `lower::Lowering::select`'s doc before assuming
/// these can collapse into one predicate vocabulary — the obvious
/// generalisation (a fire fact is just a row predicate that holds for all
/// rows or none) was implemented once and shipped a real defect, caught
/// by the live shadow comparison. `.wiki/tart/dsl.md` migration step 2
/// carries the argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// EXCLUSIVE, over the whole window: the first `Fire` arm whose
    /// predicate holds runs, and the rest do not. A `GuardPred` names a
    /// property of the FIRE, and the arm it selects is a kernel choice
    /// for the whole op list.
    Fire(model_ir::trace::GuardPred),
    /// A PARTITION: this arm covers the rows its predicate names, the
    /// rest covers the others, and BOTH run. Moving an axis from `Fire`
    /// to `Rows` is a deliberate change in what the text says, never a
    /// reinterpretation a backend performs.
    Rows(RowPred),
}

/// The arms of a [`regions`] construct.
pub struct RegionsCtx<'t> {
    guard: Option<GuardCtx>,
    rows: Option<RowsCtx<'t>>,
    t: &'t Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    out: Option<Val>,
}

impl<'t> RegionsCtx<'t> {
    /// One arm, and what runs in it.
    ///
    /// The FIRST arm fixes the construct's discipline, because the IR
    /// underneath is still two ops (`Guard` and `Peel`) and neither can
    /// express a mix. A mixed chain is a real thing to want — a fire
    /// choice inside one side of a row split — and the text already
    /// expresses it by NESTING, which is what the IR merge in migration
    /// step 6 is for. Asking for it in one flat chain is refused here
    /// rather than silently flattened into whichever op was opened first.
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

    /// Every case the arms did not name.
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

/// ONE construct for both region disciplines (`.wiki/tart/dsl.md`
/// migration step 2).
///
/// `by_rows` and `guarded_value` were two spellings of "some statements
/// run and some do not", and a reader had to know which mechanism a
/// family had reached for before they could read the arm. This is the
/// single surface: arms, each stating its own discipline, and a rest.
///
/// It lowers to today's two IR ops unchanged — a `Fire`-armed chain to
/// `Guard`, a `Rows`-armed one to `Peel` — so the goldens pin that this
/// surface changed no traced byte. Merging THOSE is migration step 6, and
/// it is a separate change with a separate gate: the IR carries the
/// discipline, so nothing here has to guess it later.
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

/// THE row-partition construct (`.wiki/tart/dsl.md` ③'s `t.by_rows`):
/// the arms' statements each cover their own rows and ALL of them run,
/// which is what separates this from the fire-level [`GuardCtx`] chain
/// (first matching arm wins, whole fire).
///
/// `shape` present makes the partition value-producing: the [`Val`] is
/// the construct's, and each region's launches bind disjoint row windows
/// of it, recording no SSA outputs of their own.
///
/// It lowers to today's [`OpKind::Peel`] — one axis word, two regions —
/// so the goldens pin that this surface changed no traced byte. What it
/// removes is the axis word from the call site: `peel` and `peel_masked`
/// were two functions naming the same mechanism over two axes, and the
/// axis is now read off the arm's predicate.
///
/// [`OpKind::Peel`]: model_ir::trace::OpKind::Peel
pub fn by_rows(
    t: &Trace,
    layer: Option<u32>,
    shape: Option<(Shape, DType)>,
    build: impl FnOnce(&mut RowsCtx<'_>),
) -> Option<Val> {
    let (idx, outs) = {
        let mut b = t.inner.borrow_mut();
        b.set_layer(layer);
        // The axis is patched at close, once the arm has named it.
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

/// [`guard`] for declarations that carry no [`M`] (the qwen3_5 bodies
/// build their own weight namespaces and run against a bare [`Trace`]):
/// the same two-way chain, opened on the tape directly.
pub fn guard_on(
    t: &Trace,
    pred: model_ir::trace::GuardPred,
    then_f: impl FnOnce(),
    else_f: impl FnOnce(),
) {
    guarded(t).arm(pred, then_f).otherwise(else_f);
}

/// Record a [`OpKind::HookSite`](model_ir::trace::OpKind::HookSite) (the HookSite slice): the layer's
/// attached programs run here at fire time, observing `q`. Since A2
/// (the class-collapse amendment) the sites live INSIDE the
/// `HasStageHooks` guard arm of the Decode/Prefill traces — the one
/// text carries them, and an unhooked fire's walk never reaches them,
/// which is the launch-list truth (the SITES' bracketing launches —
/// begin_layer, compact — exist only on hooked fires).
pub fn hook_site(stage: model_ir::trace::HookStage, q: &Val, layer: u32) {
    q.t.with(Some(layer), |b| {
        b.push_hook_site(stage, layer, q.id);
    });
}
