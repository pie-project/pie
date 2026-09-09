use std::cell::{Cell, RefCell};
use std::ops::Mul;
use std::rc::Rc;

use model_ir::{
    CacheRow, Def, Dim, Dtype, Guard, Node, Operands, Operation, Param, ParamLayout, ParamSource,
    Platform, RuntimeInput, Seam, Shard, Trace, Ty, ValueDecl, ValueId,
};

use crate::declare::Weight;
use crate::facts::Predicate;

const UNCLAIMED: u32 = u32::MAX;

#[derive(Clone)]
pub struct Recorder {
    inner: Rc<RefCell<Trace>>,

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
                drafter: None,
                seams: Vec::new(),
            })),
            at: Rc::new(Cell::new(None)),
        }
    }

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

    pub fn push(&self, op: impl Into<Operation>, ins: &[&Value]) {
        let op = op.into();
        let cond = if joins_arms(&op) {
            join(ins)
        } else {
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
            cond
        };
        let mut outs = Vec::new();
        op.outputs(&mut outs);
        let mut p = self.inner.borrow_mut();
        let index = p.nodes.len() as u32;
        for id in outs {
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

    pub fn weight(&self, w: &Weight) -> ValueId {
        let mut p = self.inner.borrow_mut();
        let w = &w.placed(p.platform);
        let mut first = None;
        for plane in w.planes() {
            let name = format!("{}{}", w.name, plane.suffix);
            let shard = restated(&w.shard, &w.shape, &plane.shape, &name);
            let index = intern(
                &mut p,
                name,
                plane.shape,
                shard,
                plane.dtype,
                w.source,
                w.layout,
            );
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
        p.values.push(ValueDecl {
            def: Def::Cache(index),
            ty: Ty::Tensor {
                shape: Vec::new(),
                dtype: Dtype::U8,
            },
        });
        ValueId((p.values.len() - 1) as u32)
    }

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

    pub fn block_drafter(&self, facts: model_ir::BlockDrafter) {
        let mut inner = self.inner.borrow_mut();
        match inner.drafter {
            Some(prior) if prior != facts => {
                panic!("the text states two block drafters: {prior:?} and then {facts:?}")
            }
            _ => inner.drafter = Some(facts),
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

    #[must_use]
    pub fn seamed(&self, name: &str, value: &Value) -> bool {
        self.inner
            .borrow()
            .seams
            .iter()
            .any(|seam| seam.seam == name && seam.values.contains(&value.id))
    }

    pub fn enter(&self, layer: u32) {
        self.at.set(Some(layer));
    }

    pub fn leave(&self) {
        self.at.set(None);
    }

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

    fn guard(&self, id: ValueId) -> Guard {
        let p = self.inner.borrow();
        match &p.values[id.0 as usize].def {
            Def::Op(i) => p
                .nodes
                .get(*i as usize)
                .map(|n| n.guard.clone())
                .unwrap_or(Guard::Always),
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
    layout: ParamLayout,
) -> u32 {
    if let Some(i) = p.params.iter().position(|q| q.name == name) {
        let seen = &p.params[i];
        assert!(
            seen.shape == shape && seen.shard == shard && seen.dtype == dtype,
            "`{name}` is declared twice with two shapes"
        );
        assert!(
            seen.source == source,
            "`{name}` is declared twice, once from the checkpoint and once as \
             a registered bank"
        );
        assert!(
            seen.layout == layout,
            "`{name}` is declared twice with two device layouts"
        );
        return i as u32;
    }
    p.params.push(Param {
        name,
        shape,
        shard,
        dtype,
        source,
        layout,
    });
    (p.params.len() - 1) as u32
}

pub(crate) fn cache_name(row: &CacheRow) -> &str {
    match row {
        CacheRow::Kv { name, .. } | CacheRow::State { name, .. } => name,
    }
}

#[derive(Clone)]
pub struct Value {
    rec: Recorder,
    id: ValueId,
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

    #[must_use]
    pub fn rows(&self) -> Dim {
        let Ty::Tensor { shape, .. } = &self.ty else {
            panic!("a struct value has no rows");
        };
        *shape
            .first()
            .unwrap_or_else(|| panic!("a rank-0 value has no rows"))
    }

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

    #[must_use]
    pub fn merge(arms: Vec<Value>) -> Value {
        assert!(arms.len() >= 2, "a merge wants at least two arms");
        let rec = arms[0].rec.clone();
        let ty = arms[0].ty.clone();
        let joined = arms.iter().map(|a| (a.id, a.cond())).collect::<Vec<_>>();
        let cond = joined
            .iter()
            .skip(1)
            .fold(joined[0].1.clone(), |c, (_, a)| Guard::or(c, a.clone()))
            .simplified();
        let arms: Vec<Guard> = joined.iter().map(|(_, c)| c.clone()).collect();
        let shared = Guard::common(&arms);
        let cond = if matches!(shared, Guard::Always) || !shared.equivalent(&cond) {
            cond
        } else {
            shared
        };
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

    pub(crate) fn under(&self, cond: Guard) -> Value {
        Value {
            rec: self.rec.clone(),
            id: self.id,
            over: Some(cond),
            ty: self.ty.clone(),
        }
    }

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
            over: Some(Guard::narrow(self.cond(), cond)),
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

pub trait Refine: Sized {
    fn refined(&self, cond: Guard) -> Self;
}

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

fn joins_arms(op: &Operation) -> bool {
    matches!(op, Operation::Attention(model_ir::Attention::Ragged { .. }))
}

fn join(ins: &[&Value]) -> Guard {
    let mut distinct: Vec<Guard> = Vec::new();
    for c in ins.iter().map(|v| v.cond()) {
        if !matches!(c, Guard::Always) && !distinct.contains(&c) {
            distinct.push(c);
        }
    }
    let Some((first, rest)) = distinct.split_first() else {
        return Guard::Always;
    };
    let joined = rest
        .iter()
        .fold(first.clone(), |a, b| Guard::or(a, b.clone()))
        .simplified();
    let shared = Guard::common(&distinct);
    if matches!(shared, Guard::Always) || !shared.equivalent(&joined) {
        joined
    } else {
        shared
    }
}

fn meet(a: Guard, b: Guard) -> Guard {
    match (a, b) {
        (Guard::Always, x) | (x, Guard::Always) => x,
        (a, _) => a,
    }
}
