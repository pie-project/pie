//! The trace machinery: a `Recorder` accumulating a `Trace`, and the `Value`
//! handles a forward pass passes around. Re-imagined from the old `record.rs`
//! for the `Def` × `Ty` world — the string `Stmt` builder is gone; wrappers
//! build typed op variants, declare their outputs with `fresh`, and `push`
//! stitches the def-use bookkeeping through the `Operands` marks.

use std::cell::{Cell, RefCell};
use std::ops::Mul;
use std::rc::Rc;

use model_ir::{
    CacheRow, Guard, Def, Dim, Dtype, Node, Operands, Operation, Param, ParamSource, Trace, Platform,
    RuntimeInput, Seam, Shard, Ty, ValueDecl, ValueId,
};

use crate::declare::Weight;
use crate::facts::Predicate;

/// A `fresh` value's node index until an op claims it. It cannot be a real
/// index, and an unclaimed sentinel survives to the validator, which names it
/// — the trace does not guess at what a wrapper forgot.
const UNCLAIMED: u32 = u32::MAX;

/// Records one forward pass. Cloned freely — every `Value` carries one — and
/// `finish` insists the trace let go of all of them.
#[derive(Clone)]
pub struct Recorder {
    inner: Rc<RefCell<Trace>>,

    /// The layer the trace is inside, driven by `enter`/`leave` (the
    /// `Layers` iterator in `forward.rs` is the usual engine).
    at: Rc<Cell<Option<u32>>>,
}

impl Recorder {
    pub(crate) fn new(name: &str, platform: Platform, caches: Vec<CacheRow>) -> Recorder {
        Recorder {
            inner: Rc::new(RefCell::new(Trace {
                name: name.to_string(),
                platform,
                params: Vec::new(),
                caches,
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
            })),
            at: Rc::new(Cell::new(None)),
        }
    }

    /// Declare a value with a placeholder def; the next `push` that lists its
    /// id among an op's outputs patches it to `Def::Op(node_index)`.
    pub fn fresh(&self, ty: Ty) -> Value {
        let mut p = self.inner.borrow_mut();
        p.values.push(ValueDecl {
            def: Def::Op(UNCLAIMED),
            ty: ty.clone(),
        });
        let id = ValueId((p.values.len() - 1) as u32);
        drop(p);
        Value {
            rec: self.clone(),
            id,
            over: None,
            ty,
        }
    }

    /// Append one node. `ins` are the condition-carrying operands: the node's
    /// guard is the meet of their conds, and mixing arms of one split is a
    /// panic here, at the line that mixed them. Outputs are read back through
    /// the `Operands` derive and their placeholder defs patched — the field
    /// marks stay the single source of truth for def-use, one more reason the
    /// derive wires the recorder rather than the wrapper doing it by hand.
    pub fn push(&self, op: impl Into<Operation>, ins: &[&Value]) {
        let op = op.into();
        let mut cond = Guard::Always;
        for v in ins {
            let c = v.cond();
            assert!(
                compatible(&cond, &c),
                "`{}` mixes values from different split arms",
                op.name(),
            );
            cond = meet(cond, c);
        }
        let mut outs = Vec::new();
        op.outputs(&mut outs);
        let mut p = self.inner.borrow_mut();
        let index = p.nodes.len() as u32;
        for id in outs {
            // Claim only placeholders. A wrapper naming a weight or another
            // node's value as its output is a fault the validator gets to
            // report in full, not something to paper over here.
            if let Some(decl) = p.values.get_mut(id.0 as usize)
                && decl.def == Def::Op(UNCLAIMED)
            {
                decl.def = Def::Op(index);
            }
        }
        p.nodes.push(Node {
            op,
            guard: cond,
            layer: self.at.get(),
        });
    }

    /// Intern a weight — one `Param` per stored plane, one `ValueId` per
    /// weight name. The value's ty is the LOGICAL tensor (all-const dims,
    /// activation dtype): what an mxfp4 bank stores is the params' business,
    /// what it multiplies as is the trace's.
    pub fn weight(&self, w: &Weight) -> ValueId {
        let mut p = self.inner.borrow_mut();
        let mut first = None;
        for plane in w.planes() {
            let name = format!("{}{}", w.name, plane.suffix);
            let shard = restated(&w.shard, &w.shape, &plane.shape, &name);
            let index = intern(&mut p, name, plane.shape, shard, plane.dtype, w.source);
            first.get_or_insert(index);
        }
        let first = first.expect("a weight stores at least one plane");
        if let Some(seen) = p.values.iter().position(|v| v.def == Def::Weight(first)) {
            return ValueId(seen as u32);
        }
        p.values.push(ValueDecl {
            def: Def::Weight(first),
            ty: Ty::Tensor {
                shape: w.shape.iter().copied().map(Dim::Const).collect(),
                dtype: w.compute_dtype(),
            },
        });
        ValueId((p.values.len() - 1) as u32)
    }

    /// The value standing for one cache space — dedup'd by index, so a layer
    /// touching its cache twice touches one id.
    pub fn cache(&self, name: &str) -> ValueId {
        let mut p = self.inner.borrow_mut();
        let index = p
            .caches
            .iter()
            .position(|row| cache_name(row) == name)
            .unwrap_or_else(|| panic!("`{name}` is not a cache the model's caches() declares"))
            as u32;
        if let Some(seen) = p.values.iter().position(|v| v.def == Def::Cache(index)) {
            return ValueId(seen as u32);
        }
        // Deliberately shapeless: a cache value is the pool pointer and
        // nothing more (design §7). Its geometry enters the graph as
        // `RuntimeInput::Geometry`, and its element layout is the cache row's
        // load-time business.
        p.values.push(ValueDecl {
            def: Def::Cache(index),
            ty: Ty::Tensor {
                shape: Vec::new(),
                dtype: Dtype::U8,
            },
        });
        ValueId((p.values.len() - 1) as u32)
    }

    /// The value the engine binds for `which`, declared once per input no
    /// matter how many layers ask.
    pub fn input(&self, which: RuntimeInput, ty: Ty) -> Value {
        let mut p = self.inner.borrow_mut();
        let id = match p.values.iter().position(|v| v.def == Def::Input(which)) {
            Some(seen) => {
                assert!(
                    p.values[seen].ty == ty,
                    "`{which:?}` is bound twice with two types",
                );
                ValueId(seen as u32)
            }
            None => {
                p.values.push(ValueDecl {
                    def: Def::Input(which),
                    ty: ty.clone(),
                });
                ValueId((p.values.len() - 1) as u32)
            }
        };
        drop(p);
        Value {
            rec: self.clone(),
            id,
            over: None,
            ty,
        }
    }

    pub fn seam(&self, name: &str, values: &[&Value]) {
        let ids = values.iter().map(|v| v.id).collect();
        self.inner.borrow_mut().seams.push(Seam {
            seam: name.to_string(),
            values: ids,
            layer: self.at.get(),
        });
    }

    pub fn enter(&self, layer: u32) {
        self.at.set(Some(layer));
    }

    pub fn leave(&self) {
        self.at.set(None);
    }

    /// Unwrap the plan and run the validator — the trace's first error
    /// surface. A bad trace panics with every fault sentence at once, so one
    /// forgotten `#[out]` reads as itself and not as ten downstream mysteries.
    pub fn finish(self) -> Trace {
        let plan = Rc::try_unwrap(self.inner)
            .unwrap_or_else(|_| panic!("a Value outlived its trace"))
            .into_inner();
        if let Err(faults) = model_ir::check(&plan) {
            let mut msg = format!("`{}` did not trace to a valid plan:", plan.name);
            for fault in &faults {
                msg.push_str("\n  ");
                msg.push_str(&fault.to_string());
            }
            panic!("{msg}");
        }
        plan
    }

    /// The guard a value settled under. Op outputs read their node's guard off
    /// the plan itself — which is why `fresh` needs no cond up front and
    /// `push` never has to reach into handles already given out.
    fn guard(&self, id: ValueId) -> Guard {
        let p = self.inner.borrow();
        match &p.values[id.0 as usize].def {
            Def::Op(i) => p
                .nodes
                .get(*i as usize)
                .map(|n| n.guard.clone())
                .unwrap_or(Guard::Always),
            // Merges carry their or-cond in the handle; inputs, weights and
            // caches are bound before the first node and guard nothing.
            _ => Guard::Always,
        }
    }
}

fn intern(
    p: &mut Trace,
    name: String,
    shape: Vec<u64>,
    shard: Shard,
    dtype: Dtype,
    source: ParamSource,
) -> u32 {
    if let Some(i) = p.params.iter().position(|q| q.name == name) {
        let seen = &p.params[i];
        assert!(
            seen.shape == shape && seen.shard == shard && seen.dtype == dtype,
            "`{name}` is declared twice with two shapes"
        );
        // Provenance is part of the declaration, not a decoration on it: the
        // same name landed once from the checkpoint and once from the serving
        // door is a plane whose contents depend on which statement ran last.
        assert!(
            seen.source == source,
            "`{name}` is declared twice, once from the checkpoint and once as \
             a registered bank"
        );
        return i as u32;
    }
    p.params.push(Param {
        name,
        shape,
        shard,
        dtype,
        source,
    });
    (p.params.len() - 1) as u32
}

pub(crate) fn cache_name(row: &CacheRow) -> &str {
    match row {
        CacheRow::Kv { name, .. } | CacheRow::State { name, .. } => name,
    }
}

/// One traced value. Carries its ty so shape queries never re-borrow the
/// plan, and reads its guard through the recorder — only a split arm
/// overrides it.
#[derive(Clone)]
pub struct Value {
    rec: Recorder,
    id: ValueId,
    /// `None` reads the producing node's guard off the plan; a split arm
    /// carries its refinement here.
    over: Option<Guard>,
    ty: Ty,
}

impl Value {
    #[must_use]
    pub fn id(&self) -> ValueId {
        self.id
    }

    #[must_use]
    pub fn rec(&self) -> &Recorder {
        &self.rec
    }

    #[must_use]
    pub fn ty(&self) -> &Ty {
        &self.ty
    }

    /// The leading dim — `Tokens` for fire-aligned activations, and what a
    /// wrapper folds `top_k` into for routed rows.
    #[must_use]
    pub fn rows(&self) -> Dim {
        let Ty::Tensor { shape, .. } = &self.ty else {
            panic!("a struct value has no rows");
        };
        *shape
            .first()
            .unwrap_or_else(|| panic!("a rank-0 value has no rows"))
    }

    /// The trailing dim, which for an activation is always a const width.
    #[must_use]
    pub fn width(&self) -> u64 {
        let Ty::Tensor { shape, .. } = &self.ty else {
            panic!("a struct value has no width");
        };
        match shape.last() {
            Some(Dim::Const(n)) => *n,
            Some(dim) => panic!("a value's trailing axis is {dim:?}, not the const a width needs"),
            None => panic!("a rank-0 value has no width"),
        }
    }

    #[must_use]
    pub fn dtype(&self) -> Dtype {
        let Ty::Tensor { dtype, .. } = &self.ty else {
            panic!("a struct value has no dtype");
        };
        *dtype
    }

    pub(crate) fn cond(&self) -> Guard {
        match &self.over {
            Some(c) => c.clone(),
            None => self.rec.guard(self.id),
        }
    }

    pub fn split<S: SplitSpec>(&self, spec: S) -> S::Arms<Value> {
        spec.arms(self)
    }

    /// The φ: a `Def::Merge` value, not an op — the compiler resolves it to
    /// slot aliasing and nothing ever dispatches it.
    #[must_use]
    pub fn merge(arms: Vec<Value>) -> Value {
        assert!(arms.len() >= 2, "a merge wants at least two arms");
        let rec = arms[0].rec.clone();
        // Arms must agree on ty; the validator's MergeArmTy rule names the
        // odd one out, so the first arm's ty stands for the merge here.
        let ty = arms[0].ty.clone();
        let joined = arms.iter().map(|a| (a.id, a.cond())).collect::<Vec<_>>();
        let cond = joined
            .iter()
            .skip(1)
            .fold(joined[0].1.clone(), |c, (_, a)| Guard::or(c, a.clone()))
            .simplified();
        let mut p = rec.inner.borrow_mut();
        p.values.push(ValueDecl {
            def: Def::Merge(joined),
            ty: ty.clone(),
        });
        let id = ValueId((p.values.len() - 1) as u32);
        drop(p);
        Value {
            rec,
            id,
            over: Some(cond),
            ty,
        }
    }

    /// **THE SAME VALUE, READ WITHOUT ITS PRODUCER'S GUARD.**
    ///
    /// A normal op output is narrow exactly where its node is: the decode
    /// attention's result exists on decode rows and nowhere else, so a
    /// consumer that took it unguarded would be reading rows nobody wrote.
    /// An IN-PLACE output under a guard is the other case, and design §8's
    /// correction class is what made it real: `linear.lora_correct`'s `y_out`
    /// aliases the `y` it adds to, so its column is written on every row of
    /// the fire — by the trunk everywhere, and by the correction again inside
    /// the adapter window. The value is defined everywhere; only the
    /// CORRECTION is narrow.
    ///
    /// Without this, the guard would leak: `residual_add(correction, y)` would
    /// take the correction's cond, the residual stream would be guarded from
    /// that layer on, and the next layer's split would refuse to mix with it.
    /// With it, the model text reads the way the mechanism works — one
    /// statement, no merge, no arm.
    ///
    /// `model_ir::check::classes` is the other half: its demand walk follows
    /// the alias when the node is not live, so the classes that skip the
    /// correction still demand the trunk that filled the column.
    #[must_use]
    pub fn everywhere(&self) -> Value {
        Value {
            rec: self.rec.clone(),
            id: self.id,
            over: Some(Guard::Always),
            ty: self.ty.clone(),
        }
    }
}

impl Refine for Value {
    fn refined(&self, cond: Guard) -> Value {
        Value {
            rec: self.rec.clone(),
            id: self.id,
            over: Some(Guard::and(self.cond(), cond)),
            ty: self.ty.clone(),
        }
    }
}

impl Mul<f32> for Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Value {
        let y = self.rec.fresh(self.ty.clone());
        self.rec.push(
            model_ir::Elementwise::MulScalar {
                s: rhs,
                x: self.id,
                x_out: y.id(),
            },
            &[&self],
        );
        y
    }
}

/// What a split hands out arms of. One algorithm, two carriers: a [`Value`]
/// arm is the same traced value read under a narrower guard, and an
/// [`Input`](crate::Input) arm is the same handle whose every value comes out
/// under that guard. Cut by the same spec they carry structurally equal conds,
/// which is exactly what [`Recorder::push`]'s compatibility check compares —
/// so a schedule built off one class's arm and a query read off another's is
/// refused at the line that mixed them.
pub trait Refine: Sized {
    fn refined(&self, cond: Guard) -> Self;
}

/// How a spec cuts a carrier into arms: `&Predicate` gives the two-way
/// yes/no pair, `[Predicate; N]` the priority-ordered n-way carving. The
/// arm computation is written once, over any [`Refine`].
pub trait SplitSpec {
    type Arms<T>;
    fn arms<T: Refine>(self, of: &T) -> Self::Arms<T>;
}

impl SplitSpec for &Predicate {
    type Arms<T> = (T, T);

    fn arms<T: Refine>(self, of: &T) -> (T, T) {
        let c = cond_of(self);
        (of.refined(c.clone()), of.refined(Guard::not(c)))
    }
}

impl<const N: usize> SplitSpec for [Predicate; N] {
    type Arms<T> = [T; N];

    fn arms<T: Refine>(self, of: &T) -> [T; N] {
        let mut not_prior = Guard::Always;
        self.each_ref().map(|p| {
            let mine = match p {
                Predicate::Rest => not_prior.clone(),
                p => {
                    let c = cond_of(p);
                    let mine = Guard::and(not_prior.clone(), c.clone());
                    not_prior = Guard::and(not_prior.clone(), Guard::not(c));
                    mine
                }
            };
            of.refined(mine)
        })
    }
}

/// A plane of a stored shard restated against what the plane actually holds:
/// an mxfp4 codes plane cuts by blocks, not logical columns.
fn restated(shard: &Shard, logical: &[u64], plane: &[u64], name: &str) -> Shard {
    let Shard::Cut { axis, segments } = shard else {
        return Shard::Replicated;
    };
    let at = *axis as usize;
    let whole = logical[at];
    let stored = *plane.get(at).unwrap_or_else(|| {
        panic!("`{name}` stores {plane:?} and its cut names axis {at} of {logical:?}")
    });
    assert_eq!(
        logical[..at],
        plane[..at],
        "`{name}`: the axes before its cut are not stored verbatim",
    );
    assert!(
        stored > 0 && whole.is_multiple_of(stored),
        "`{name}`: axis {at} is {whole} logically and {stored} as stored",
    );
    let block = whole / stored;
    Shard::Cut {
        axis: *axis,
        segments: segments
            .iter()
            .map(|s| {
                assert!(
                    s.is_multiple_of(block),
                    "`{name}`: a segment of {s} is not a whole number of \
                     {block}-wide blocks",
                );
                s / block
            })
            .collect(),
    }
}

fn cond_of(p: &Predicate) -> Guard {
    match p {
        Predicate::Fact { bit } => Guard::Fact(*bit),
        Predicate::Not(a) => Guard::not(cond_of(a)),
        Predicate::And(a, b) => Guard::and(cond_of(a), cond_of(b)),
        Predicate::Rest => panic!("Predicate::rest() belongs only in an n-way split"),
    }
}

fn compatible(a: &Guard, b: &Guard) -> bool {
    matches!(a, Guard::Always) || matches!(b, Guard::Always) || a == b
}

fn meet(a: Guard, b: Guard) -> Guard {
    match (a, b) {
        (Guard::Always, x) | (x, Guard::Always) => x,
        (a, _) => a,
    }
}
