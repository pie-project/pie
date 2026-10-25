//! Forms a backend's fused kernels from the primitive ops a model writes.
//!
//! Each rule is a pattern written by example with the model-writing ops
//! (`poem::pattern`) and the fused op that replaces a match of it. A rule is
//! built by its constructor in [`rules`], parameterised by the forms it
//! forms (`rules::matmul_glu(&[GluAct::Geglu])`). A backend owns the rules
//! it implements, so a rule it holds forms only ops its dispatch fires; the
//! compiler runs those rules alone, in the order [`rules::all`] gives, and
//! every match is replaced.

pub mod chain;
pub mod flags;
pub mod rules;
mod search;

use std::sync::LazyLock;

use poem::pattern::Template;
use poem_ir::{Dim, Dtype, Fused, Operation, Trace, Ty, ValueId};

use search::{Binding, Shape};

/// One fusion: the fused op it forms (`kernel`, its label), the patterns it
/// matches and how a match is replaced.
pub struct Rule {
    kernel: &'static str,
    /// Which of its op's forms the rule forms, where it forms one of several
    /// (`""` where its op has one): the rules of one op may run apart.
    form: &'static str,
    shapes: Vec<Shape>,
    when: fn(&Match) -> bool,
    replace: fn(&Match) -> Fused,
}

impl Rule {
    /// The label of the fused op this rule forms; its profile flag is
    /// `fuse.<name>`.
    #[must_use]
    pub fn kernel(&self) -> &'static str {
        self.kernel
    }

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
            form: "",
            shapes,
            when: |_| true,
            replace,
        }
    }

    /// The rule forms its op's `form`; it runs at that form's place.
    fn form(mut self, form: &'static str) -> Rule {
        self.form = form;
        self
    }

    fn when(mut self, when: fn(&Match) -> bool) -> Rule {
        self.when = when;
        self
    }
}

/// One match of a rule's pattern, as the replacement reads it: the trace's
/// values and ops bound to the pattern's names.
struct Match<'a> {
    trace: &'a Trace,
    shape: &'a Shape,
    binding: &'a Binding,
}

impl Match<'_> {
    /// The trace value bound to `name`, if this alternative names it.
    fn get(&self, name: &str) -> Option<ValueId> {
        let p = self.shape.template.value(name)?;
        self.binding.values[p.0 as usize]
    }

    fn value(&self, name: &str) -> ValueId {
        self.get(name)
            .unwrap_or_else(|| panic!("the pattern binds no value `{name}`"))
    }

    /// The op that computes the value bound to `name`.
    fn op(&self, name: &str) -> &Operation {
        let v = self.value(name);
        match self.trace.values[v.0 as usize].def {
            poem_ir::Def::Op(i) => &self.trace.nodes[i as usize].op,
            _ => panic!("`{name}` is a hole, not a value the pattern computes"),
        }
    }

    fn attr(&self, name: &str) -> Option<&crate::tree::Tree> {
        let at = self
            .shape
            .template
            .attrs
            .iter()
            .position(|(n, _)| *n == name)?;
        self.binding.attrs[at].as_ref()
    }

    fn f32(&self, name: &str) -> f32 {
        match self.attr(name) {
            Some(crate::tree::Tree::F32(bits)) => f32::from_bits(*bits),
            other => panic!("`{name}` binds {other:?}, not an f32"),
        }
    }

    fn get_f32(&self, name: &str) -> Option<f32> {
        match self.attr(name)? {
            crate::tree::Tree::F32(bits) => Some(f32::from_bits(*bits)),
            _ => None,
        }
    }

    fn u32(&self, name: &str) -> u32 {
        self.get_u32(name)
            .unwrap_or_else(|| panic!("the pattern binds no u32 `{name}`"))
    }

    fn get_u32(&self, name: &str) -> Option<u32> {
        match self.attr(name)? {
            crate::tree::Tree::Int(n) => u32::try_from(*n).ok(),
            _ => None,
        }
    }

    fn ty(&self, name: &str) -> &Ty {
        &self.trace.values[self.value(name).0 as usize].ty
    }

    fn width(&self, name: &str) -> Option<u64> {
        match self.ty(name) {
            Ty::Tensor { shape, .. } => match shape.last() {
                Some(Dim::Const(width)) => Some(*width),
                _ => None,
            },
            Ty::Struct(_) => None,
        }
    }

    /// The dtype the weight bound to `name` is stored in (a packed bank's
    /// own, where its value's type is the element it reads as).
    fn weight_dtype(&self, name: &str) -> Option<Dtype> {
        match self.trace.values[self.value(name).0 as usize].def {
            poem_ir::Def::Weight(i) => Some(self.trace.params[i as usize].dtype),
            _ => None,
        }
    }

    fn dtype(&self, name: &str) -> Option<Dtype> {
        match self.ty(name) {
            Ty::Tensor { dtype, .. } => Some(*dtype),
            Ty::Struct(_) => None,
        }
    }
}

/// Every rule's op and form in the order rules run: a rule that forms a
/// longer chain runs before one that forms a piece of it.
static ORDER: LazyLock<Vec<(&'static str, &'static str)>> = LazyLock::new(|| {
    rules::all()
        .iter()
        .map(|rule| (rule.kernel, rule.form))
        .collect()
});

/// `trace` with every match of `rules` replaced by its fused op. The rules
/// run in [`rules::all`]'s order whatever order they are given in.
#[must_use]
pub fn fuse<'r>(mut trace: Trace, rules: impl IntoIterator<Item = &'r Rule>) -> Trace {
    let rank = |rule: &&Rule| {
        ORDER
            .iter()
            .position(|at| *at == (rule.kernel, rule.form))
            .unwrap_or_else(|| panic!("`{}` ({}) is a form no rule forms", rule.kernel, rule.form))
    };
    let mut ordered: Vec<&Rule> = rules.into_iter().collect();
    ordered.sort_by_key(rank);
    for rule in ordered {
        trace = search::apply(trace, rule);
    }
    search::compact(trace)
}
