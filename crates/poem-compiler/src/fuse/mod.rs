//! Forms a backend's fused kernels from the primitive ops a model writes.
//!
//! Each rule is a pattern written by example with the model-writing ops
//! (`poem_dsl::pattern`) and the fused op that replaces a match of it. A
//! backend lists the fused kernels it ships; only the rules that form one of
//! those run, and every match is replaced.

mod rules;
mod search;
mod tree;

use std::sync::LazyLock;

use poem_dsl::pattern::Template;
use poem_ir::{Dim, Dtype, Fused, Operation, Trace, Ty, ValueId};

use search::{Binding, Shape};

pub struct Rule {
    pub kernel: &'static str,
    shapes: Vec<Shape>,
    when: fn(&Match) -> bool,
    replace: fn(&Match) -> Fused,
}

impl Rule {
    /// A rule whose alternatives are tried in order at each op.
    fn new(
        kernel: &'static str,
        alternatives: impl IntoIterator<Item = Template>,
        replace: fn(&Match) -> Fused,
    ) -> Rule {
        let shapes: Vec<Shape> = alternatives.into_iter().map(Shape::new).collect();
        assert!(!shapes.is_empty(), "`{kernel}` states no pattern");
        Rule {
            kernel,
            shapes,
            when: |_| true,
            replace,
        }
    }

    fn when(mut self, when: fn(&Match) -> bool) -> Rule {
        self.when = when;
        self
    }
}

/// One match of a rule's pattern, as the replacement reads it: the trace's
/// values and ops bound to the pattern's names.
pub struct Match<'a> {
    trace: &'a Trace,
    shape: &'a Shape,
    binding: &'a Binding,
}

impl Match<'_> {
    /// The trace value bound to `name`, if this alternative names it.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<ValueId> {
        let p = self.shape.template.value(name)?;
        self.binding.values[p.0 as usize]
    }

    #[must_use]
    pub fn value(&self, name: &str) -> ValueId {
        self.get(name)
            .unwrap_or_else(|| panic!("the pattern binds no value `{name}`"))
    }

    /// The op that computes the value bound to `name`.
    #[must_use]
    pub fn op(&self, name: &str) -> &Operation {
        let v = self.value(name);
        match self.trace.values[v.0 as usize].def {
            poem_ir::Def::Op(i) => &self.trace.nodes[i as usize].op,
            _ => panic!("`{name}` is a hole, not a value the pattern computes"),
        }
    }

    fn attr(&self, name: &str) -> Option<&tree::Tree> {
        let at = self
            .shape
            .template
            .attrs
            .iter()
            .position(|(n, _)| *n == name)?;
        self.binding.attrs[at].as_ref()
    }

    #[must_use]
    pub fn f32(&self, name: &str) -> f32 {
        match self.attr(name) {
            Some(tree::Tree::F32(bits)) => f32::from_bits(*bits),
            other => panic!("`{name}` binds {other:?}, not an f32"),
        }
    }

    #[must_use]
    pub fn u32(&self, name: &str) -> u32 {
        self.get_u32(name)
            .unwrap_or_else(|| panic!("the pattern binds no u32 `{name}`"))
    }

    #[must_use]
    pub fn get_u32(&self, name: &str) -> Option<u32> {
        match self.attr(name)? {
            tree::Tree::Int(n) => u32::try_from(*n).ok(),
            _ => None,
        }
    }

    fn ty(&self, name: &str) -> &Ty {
        &self.trace.values[self.value(name).0 as usize].ty
    }

    #[must_use]
    pub fn width(&self, name: &str) -> Option<u64> {
        match self.ty(name) {
            Ty::Tensor { shape, .. } => match shape.last() {
                Some(Dim::Const(width)) => Some(*width),
                _ => None,
            },
            Ty::Struct(_) => None,
        }
    }

    #[must_use]
    pub fn dtype(&self, name: &str) -> Option<Dtype> {
        match self.ty(name) {
            Ty::Tensor { dtype, .. } => Some(*dtype),
            Ty::Struct(_) => None,
        }
    }
}

static RULES: LazyLock<Vec<Rule>> = LazyLock::new(rules::all);

/// Every fused kernel some rule forms, in the order the rules run.
pub fn kernels() -> impl Iterator<Item = &'static str> {
    RULES.iter().map(|rule| rule.kernel)
}

/// `trace` with every match of the rules that form one of `kernels` replaced
/// by its fused op.
#[must_use]
pub fn fuse(mut trace: Trace, kernels: &[&str]) -> Trace {
    for kernel in kernels {
        assert!(
            RULES.iter().any(|rule| rule.kernel == *kernel),
            "`{kernel}` is a fused kernel no rule forms"
        );
    }
    for rule in RULES.iter().filter(|rule| kernels.contains(&rule.kernel)) {
        trace = search::apply(trace, rule);
    }
    search::compact(trace)
}
