//! The supergraph recorder: the text runs once, conditions ride as data.

use std::cell::RefCell;
use std::ops::Mul;
use std::rc::Rc;

use model_ir::plan::{CacheRow, Cond, Op, Param, Plan, Seam, Shard, ValueDef, ValueId};

use crate::axes::Dtype;
use crate::declare::{Norm, Tensor};
use crate::facts::Predicate;
use crate::Plane;

#[derive(Clone)]
pub struct Recorder {
    inner: Rc<RefCell<Plan>>,
}

impl Recorder {
    pub(crate) fn new(name: &str, plane: Plane, facts: &[&str], caches: Vec<CacheRow>) -> Recorder {
        Recorder {
            inner: Rc::new(RefCell::new(Plan {
                name: name.to_string(),
                plane,
                facts: facts.iter().map(|f| (*f).to_string()).collect(),
                params: Vec::new(),
                caches,
                values: Vec::new(),
                ops: Vec::new(),
                seams: Vec::new(),
            })),
        }
    }

    pub(crate) fn finish(self) -> Plan {
        Rc::try_unwrap(self.inner)
            .unwrap_or_else(|_| panic!("a Value outlived its trace"))
            .into_inner()
    }

    pub(crate) fn runtime(&self, name: &str) -> Value {
        let mut p = self.inner.borrow_mut();
        let id = p
            .values
            .iter()
            .position(|v| matches!(v, ValueDef::Runtime(n) if n == name))
            .unwrap_or_else(|| {
                p.values.push(ValueDef::Runtime(name.to_string()));
                p.values.len() - 1
            }) as ValueId;
        Value {
            rec: self.clone(),
            id,
            cond: Cond::Always,
        }
    }

    pub(crate) fn param<W: Dtype>(&self, t: &Tensor<W>) {
        self.plane(&t.name, &t.shape, &t.shard, W::NAME);
    }

    /// One parameter row, at a name, shape and repr the caller has already
    /// decided.
    ///
    /// [`Recorder::param`] is the dense reading of this and passes the
    /// tensor's own three columns straight through; a bank's planes differ
    /// from the tensor in two of the three, which is what this exists for.
    /// The SHARD is not one of them — every plane of one bank is cut the same
    /// way, because they are the same weight described twice.
    pub(crate) fn plane(&self, name: &str, shape: &[u64], shard: &Shard, repr: &str) {
        let mut p = self.inner.borrow_mut();
        if let Some(seen) = p.params.iter().find(|q| q.name == name) {
            assert!(
                seen.shape == shape && &seen.shard == shard,
                "`{name}` is declared twice with two shapes"
            );
            return;
        }
        p.params.push(Param {
            name: name.to_string(),
            shape: shape.to_vec(),
            shard: shard.clone(),
            repr: repr.to_string(),
        });
    }

    pub(crate) fn seam(&self, seam: &str, values: &[&Value], layer: Option<u32>) {
        let ids = values.iter().map(|v| v.id).collect();
        self.inner.borrow_mut().seams.push(Seam {
            seam: seam.to_string(),
            values: ids,
            layer,
        });
    }
}

/// One statement under construction; every role fn builds one.
pub(crate) struct Stmt<'r> {
    rec: &'r Recorder,
    kernel: &'static str,
    inputs: Vec<ValueId>,
    weights: Vec<String>,
    params: Vec<u64>,
    cache: Option<String>,
    layer: Option<u32>,
    cond: Cond,
}

impl<'r> Stmt<'r> {
    pub(crate) fn value(mut self, v: &Value) -> Self {
        assert!(
            compatible(&self.cond, &v.cond),
            "`{}` mixes values from different split arms",
            self.kernel
        );
        self.cond = meet(self.cond, v.cond.clone());
        self.inputs.push(v.id);
        self
    }

    pub(crate) fn weight<W: Dtype>(mut self, t: &Tensor<W>) -> Self {
        self.rec.param(t);
        self.weights.push(t.name.clone());
        self
    }

    /// A QUANTISED bank: one thing a text names, however many parameters the
    /// repr stores it as.
    ///
    /// `weight` above is this with the plane count pinned at one, and that
    /// is not a coincidence — it is the same slot with a repr whose storage
    /// is its logical shape. What the two differ in is what the DECLARATION
    /// said: a `Const<Self::Tensor<T>>` slot promises one rectangle of
    /// elements, a `Const<Self::Bank<R>>` slot promises `R::PLANES` planes
    /// of bytes, and the columns a statement records have to match the slot
    /// the point declared or the dispatch reads the wrong one.
    ///
    /// THE PLANE ORDER IS THE CONTRACT. `Dtype::planes` states it, this
    /// records it, `BoundOp::bank` reads it back positionally, and nothing
    /// in between re-derives it from a name.
    pub(crate) fn bank<W: Dtype>(mut self, t: &Tensor<W>) -> Self {
        for plane in W::planes(&t.shape) {
            let name = format!("{}{}", t.name, plane.suffix);
            self.rec.plane(&name, &plane.shape, &t.shard, plane.repr);
            self.weights.push(name);
        }
        self
    }

    pub(crate) fn norm<W: Dtype>(self, n: &Norm<W>) -> Self {
        let eps = n.eps;
        self.weight(&n.weight).float(eps)
    }

    pub(crate) fn int(mut self, v: u32) -> Self {
        self.params.push(u64::from(v));
        self
    }

    pub(crate) fn float(mut self, v: f32) -> Self {
        self.params.push(u64::from(v.to_bits()));
        self
    }

    pub(crate) fn window(self, w: Option<u32>) -> Self {
        self.int(w.unwrap_or(0))
    }

    pub(crate) fn cache(mut self, name: &str) -> Self {
        self.cache = Some(name.to_string());
        self
    }

    pub(crate) fn layer(mut self, l: u32) -> Self {
        self.layer = Some(l);
        self
    }

    pub(crate) fn done(self) -> Value {
        let (rec, cond) = (self.rec.clone(), self.cond.clone());
        let id = self.push(1)[0];
        Value { rec, id, cond }
    }

    pub(crate) fn pair(self) -> (Value, Value) {
        let (rec, cond) = (self.rec.clone(), self.cond.clone());
        let ids = self.push(2);
        (
            Value { rec: rec.clone(), id: ids[0], cond: cond.clone() },
            Value { rec, id: ids[1], cond },
        )
    }

    pub(crate) fn triple(self) -> (Value, Value, Value) {
        let (rec, cond) = (self.rec.clone(), self.cond.clone());
        let ids = self.push(3);
        (
            Value { rec: rec.clone(), id: ids[0], cond: cond.clone() },
            Value { rec: rec.clone(), id: ids[1], cond: cond.clone() },
            Value { rec, id: ids[2], cond },
        )
    }

    pub(crate) fn effect(self) {
        self.push(0);
    }

    fn push(self, outputs: usize) -> Vec<ValueId> {
        let mut p = self.rec.inner.borrow_mut();
        let op_index = p.ops.len() as u32;
        let ids: Vec<ValueId> = (0..outputs)
            .map(|_| {
                p.values.push(ValueDef::Stmt(op_index));
                (p.values.len() - 1) as ValueId
            })
            .collect();
        p.ops.push(Op {
            kernel: self.kernel.to_string(),
            inputs: self.inputs,
            outputs: ids.clone(),
            weights: self.weights,
            params: self.params,
            cache: self.cache,
            layer: self.layer,
            cond: self.cond,
        });
        ids
    }
}

#[derive(Clone)]
pub struct Value {
    pub(crate) rec: Recorder,
    pub(crate) id: ValueId,
    pub(crate) cond: Cond,
}

impl Value {
    pub(crate) fn stmt(&self, kernel: &'static str) -> Stmt<'_> {
        Stmt {
            rec: &self.rec,
            kernel,
            inputs: Vec::new(),
            weights: Vec::new(),
            params: Vec::new(),
            cache: None,
            layer: None,
            cond: Cond::Always,
        }
        .value(self)
    }

    pub fn split<S: SplitSpec>(&self, spec: S) -> S::Arms {
        spec.arms(self)
    }

    #[must_use]
    pub fn merge(arms: Vec<Value>) -> Value {
        assert!(arms.len() >= 2, "merge! wants at least two arms");
        let rec = arms[0].rec.clone();
        let joined = arms
            .iter()
            .map(|a| (a.id, a.cond.clone()))
            .collect::<Vec<_>>();
        let cond = arms
            .iter()
            .skip(1)
            .fold(arms[0].cond.clone(), |c, a| Cond::or(c, a.cond.clone()))
            .simplified();
        let mut p = rec.inner.borrow_mut();
        p.values.push(ValueDef::Merge(joined));
        let id = (p.values.len() - 1) as ValueId;
        drop(p);
        Value { rec, id, cond }
    }

    fn refined(&self, cond: Cond) -> Value {
        Value {
            rec: self.rec.clone(),
            id: self.id,
            cond: Cond::and(self.cond.clone(), cond),
        }
    }
}

impl Mul<f32> for Value {
    type Output = Value;

    fn mul(self, rhs: f32) -> Value {
        self.stmt("norm.mul_scalar").float(rhs).done()
    }
}

/// A token-rowed stream paired with the fire's request boundaries.
#[derive(Clone)]
pub struct Windows {
    pub data: Value,
    pub indptr: Value,
}

pub trait SplitSpec {
    type Arms;
    fn arms(self, v: &Value) -> Self::Arms;
}

impl SplitSpec for &Predicate {
    type Arms = (Value, Value);

    fn arms(self, v: &Value) -> (Value, Value) {
        let c = cond_of(self);
        (v.refined(c.clone()), v.refined(Cond::not(c)))
    }
}

impl<const N: usize> SplitSpec for [Predicate; N] {
    type Arms = [Value; N];

    fn arms(self, v: &Value) -> [Value; N] {
        let mut not_prior = Cond::Always;
        self.each_ref().map(|p| {
            let mine = match p {
                Predicate::Rest => not_prior.clone(),
                p => {
                    let c = cond_of(p);
                    let mine = Cond::and(not_prior.clone(), c.clone());
                    not_prior = Cond::and(not_prior.clone(), Cond::not(c));
                    mine
                }
            };
            v.refined(mine)
        })
    }
}

fn cond_of(p: &Predicate) -> Cond {
    match p {
        Predicate::Fact { bit, .. } => Cond::Fact(*bit),
        Predicate::Not(a) => Cond::not(cond_of(a)),
        Predicate::And(a, b) => Cond::and(cond_of(a), cond_of(b)),
        Predicate::Rest => panic!("Predicate::rest() belongs only in an n-way split"),
    }
}

fn compatible(a: &Cond, b: &Cond) -> bool {
    matches!(a, Cond::Always) || matches!(b, Cond::Always) || a == b
}

fn meet(a: Cond, b: Cond) -> Cond {
    match (a, b) {
        (Cond::Always, x) | (x, Cond::Always) => x,
        (a, _) => a,
    }
}

#[macro_export]
macro_rules! merge {
    ($($arm:expr),+ $(,)?) => {
        $crate::Value::merge(vec![$($arm),+])
    };
}
