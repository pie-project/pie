//! Fusion patterns written by example: a pattern is a small trace of the ops a
//! model writes, over named holes, which the compiler matches against a model's
//! trace and replaces with one fused op.

use std::cell::RefCell;

use poem_ir::{CacheRow, Def, Dim, Dtype, Platform, Trace, Ty, ValueId};

use crate::declare::Weight;
use crate::record::{Recorder, Value};

/// A wildcard attribute. The pattern traces with a sentinel no model states
/// (a NaN payload, a width beyond any device), and the matcher binds whatever
/// the matched op holds in its place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attr {
    F32(u32),
    U32(u32),
}

impl Attr {
    const F32_BASE: u32 = 0x7fc0_5000;
    const U32_BASE: u32 = 0x7a5a_0000;
}

#[derive(Clone, Debug)]
pub struct Template {
    pub trace: Trace,
    pub names: Vec<(&'static str, ValueId)>,
    pub exports: Vec<ValueId>,
    pub attrs: Vec<(&'static str, Attr)>,
    pub free: Vec<&'static str>,
}

impl Template {
    #[must_use]
    pub fn value(&self, name: &str) -> Option<ValueId> {
        self.names.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }
}

pub struct Pattern {
    rec: Recorder,
    parts: RefCell<Parts>,
}

#[derive(Default)]
struct Parts {
    names: Vec<(&'static str, ValueId)>,
    exports: Vec<ValueId>,
    attrs: Vec<(&'static str, Attr)>,
    free: Vec<&'static str>,
}

impl Pattern {
    fn named(&self, name: &'static str, id: ValueId) {
        let mut parts = self.parts.borrow_mut();
        assert!(
            parts.names.iter().all(|(n, _)| *n != name),
            "`{name}` names two values of one pattern"
        );
        parts.names.push((name, id));
    }

    /// A hole of the given type: it matches any value the trace holds there.
    pub fn value(&self, name: &'static str, ty: Ty) -> Value {
        let v = self.rec.fresh(ty);
        self.named(name, v.id());
        v
    }

    /// A hole over rows of `width` bf16 columns.
    pub fn rows(&self, name: &'static str, width: u64) -> Value {
        self.value(
            name,
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Const(width)],
                dtype: Dtype::Bf16,
            },
        )
    }

    /// A hole of one i32 per row: positions, token ids, a row's lane.
    pub fn indices(&self, name: &'static str) -> Value {
        self.value(
            name,
            Ty::Tensor {
                shape: vec![Dim::Tokens],
                dtype: Dtype::I32,
            },
        )
    }

    /// A hole that matches only a weight.
    pub fn weight(&self, name: &'static str, shape: impl IntoIterator<Item = u64>) -> Weight {
        let w = Weight::sym(format!("pattern.{name}"), shape, Dtype::Bf16);
        let id = self.rec.weight(&w);
        self.named(name, id);
        w
    }

    /// A hole that matches only a cache.
    pub fn cache(&self, name: &'static str) -> ValueId {
        let row = format!("pattern.{name}");
        self.rec.declare_cache(CacheRow::Kv {
            name: row.clone(),
            planes: Vec::new(),
            dtype: Dtype::Bf16,
            space: 0,
            window: None,
            head_dim: 0,
            shard: poem_ir::Shard::Replicated,
        });
        let id = self.rec.cache(&row);
        self.named(name, id);
        id
    }

    pub fn f32(&self, name: &'static str) -> f32 {
        let bits = Attr::F32_BASE + self.attr_count();
        self.parts.borrow_mut().attrs.push((name, Attr::F32(bits)));
        f32::from_bits(bits)
    }

    pub fn u32(&self, name: &'static str) -> u32 {
        let value = Attr::U32_BASE + self.attr_count();
        self.parts.borrow_mut().attrs.push((name, Attr::U32(value)));
        value
    }

    fn attr_count(&self) -> u32 {
        self.parts.borrow().attrs.len() as u32
    }

    /// Every op of the pattern matches whatever this field holds; the
    /// replacement reads it off the matched op.
    pub fn free(&self, field: &'static str) {
        self.parts.borrow_mut().free.push(field);
    }

    /// Names a value the pattern computes, for the replacement to read; a
    /// value only named must have no reader outside the match.
    pub fn name(&self, name: &'static str, v: &Value) {
        self.named(name, v.id());
    }

    /// Names a value the fused op still writes, which the rest of the trace
    /// may read.
    pub fn export(&self, name: &'static str, v: &Value) {
        self.named(name, v.id());
        self.parts.borrow_mut().exports.push(v.id());
    }
}

pub fn template(build: impl FnOnce(&Pattern)) -> Template {
    let p = Pattern {
        rec: Recorder::new("pattern", Platform::Cuda, Vec::new()),
        parts: RefCell::new(Parts::default()),
    };
    build(&p);
    let Pattern { rec, parts } = p;
    let parts = parts.into_inner();
    let trace = rec.finish_unchecked();
    for &id in &parts.exports {
        assert!(
            matches!(trace.values[id.0 as usize].def, Def::Op(i) if (i as usize) < trace.nodes.len()),
            "a pattern exports only a value it computes"
        );
    }
    Template {
        trace,
        names: parts.names,
        exports: parts.exports,
        attrs: parts.attrs,
        free: parts.free,
    }
}
