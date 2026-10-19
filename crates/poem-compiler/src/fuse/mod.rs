//! Forms a backend's fused kernels from the primitive ops a model writes.
//!
//! Each rule is a pattern written by example with the model-writing ops
//! (`poem::pattern`) and the fused op that replaces a match of it. A
//! backend lists the fused kernels it ships; only the rules that form one of
//! those run, and every match is replaced.

pub mod flags;
mod rules;
mod search;

use std::sync::LazyLock;

use poem::pattern::Template;
use poem_ir::{Dim, Dtype, Fused, Operation, Trace, Ty, ValueId};

use search::{Binding, Shape};

struct Rule {
    kernel: &'static str,
    shapes: Vec<Shape>,
    when: fn(&Match) -> bool,
    replace: fn(&Match) -> Fused,
    /// Whether a seam no shell reads (a marker such as `attn.qv`, not one of
    /// `EXPORT_SEAMS`) may hold a value this rule folds away. Every rule
    /// keeps every seam's values but the ones that say otherwise.
    through_markers: bool,
    /// Whether the fused op lands where the first matched op was, not the
    /// last: for matches whose later ops' outputs are read in between (two
    /// norms of one row, the first's reader ahead of the second).
    land_first: bool,
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
            through_markers: false,
            land_first: false,
        }
    }

    /// A marker seam (one no shell reads) does not keep this rule from
    /// folding the value it marks.
    fn through_markers(mut self) -> Rule {
        self.through_markers = true;
        self
    }

    fn land_first(mut self) -> Rule {
        self.land_first = true;
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
