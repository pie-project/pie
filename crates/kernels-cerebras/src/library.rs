//! The CSL kernel library: every op one function over pointers and runtime
//! sizes, so a program's phases are calls and its code size does not grow
//! with the number of ops in a fire. Each function (with the helpers only it
//! uses) is one file, `kernels/<group>/<name>.csl`, called as `k_<name>`; a
//! program carries only the units its phases call, since a PE holds 48 KB of
//! code and data together. CSL has no namespaces and a program is one text,
//! so every name is unique across the tree.
//!
//! `build.rs` takes the census: the units, the functions each declares with
//! their parameter formats, the units each calls into, and a [`Kernel`] id
//! for every unit entry and every `prim/` function.

include!(concat!(env!("OUT_DIR"), "/units.rs"));

/// One library file.
#[derive(Debug)]
pub struct Unit {
    /// `elemwise/rmsnorm`.
    pub path: &'static str,
    /// The function it is called by (`k_rmsnorm`); a `prim/` unit is a
    /// helper set and its name only keys it.
    pub name: &'static str,
    pub text: &'static str,
    /// The units it calls into (indices into [`UNITS`]), the prelude aside.
    pub deps: &'static [usize],
    /// The kernels it declares: its entry, or every function of a `prim/`
    /// unit.
    pub fns: &'static [Function],
}

/// A function a unit declares.
#[derive(Debug)]
pub struct Function {
    pub name: &'static str,
    pub params: &'static [Param],
}

/// A parameter's format: how an op row's word becomes the argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    PtrF32,
    PtrI32,
    PtrU32,
    PtrU64,
    I32,
    U32,
    F32,
    Bool,
}

impl Kernel {
    /// The CSL function (`k_rmsnorm`).
    #[must_use]
    pub fn name(self) -> &'static str {
        self.function().name
    }

    #[must_use]
    pub fn params(self) -> &'static [Param] {
        self.function().params
    }

    /// The unit declaring it.
    #[must_use]
    pub fn unit(self) -> &'static Unit {
        &UNITS[self.at().0]
    }

    fn function(self) -> &'static Function {
        let (u, f) = self.at();
        &UNITS[u].fns[f]
    }

    /// The kernel called `name`.
    #[must_use]
    pub fn of(name: &str) -> Option<Kernel> {
        Kernel::ALL.iter().copied().find(|k| k.name() == name)
    }
}

impl Unit {
    /// This unit and every unit it calls into, transitively.
    pub fn with_deps(&'static self) -> impl Iterator<Item = &'static Unit> {
        let mut order = vec![self];
        let mut at = 0;
        while at < order.len() {
            for d in order[at].deps {
                let d = &UNITS[*d];
                if !order.iter().any(|u| std::ptr::eq(*u, d)) {
                    order.push(d);
                }
            }
            at += 1;
        }
        order.into_iter()
    }
}

/// The prelude every program carries: base DSDs, the dot reducer, copies.
#[must_use]
pub fn prelude() -> &'static str {
    UNITS
        .iter()
        .find(|u| u.path == "prim/prelude")
        .expect("the prelude unit")
        .text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_census_reads_every_kernel() {
        assert!(prelude().contains("fn k_round_bf16"));
        use Param::*;
        assert_eq!(
            Kernel::Matmul.params(),
            [PtrF32, PtrF32, PtrF32, I32, I32, I32]
        );
        assert_eq!(Kernel::RoundBf16.params(), [PtrF32, I32]);
        assert_eq!(Kernel::RoundBf16.unit().path, "prim/prelude");
        assert!(Kernel::Attend.params().len() > 30);
        assert_eq!(Kernel::of("k_matmul"), Some(Kernel::Matmul));
        assert!(Kernel::of("nope").is_none());
        let u = Kernel::Matmul.unit();
        assert_eq!((u.path, u.name), ("linear/matmul", "k_matmul"));
        assert!(u.text.starts_with("// y[m, n]"));
        // A unit's helpers are no kernels; prim functions are.
        assert!(Kernel::of("k_sig_mul").is_none());
        assert_eq!(Kernel::of("k_unpack_bf16"), Some(Kernel::UnpackBf16));
    }

    #[test]
    fn a_unit_brings_the_units_it_calls() {
        let paths = |k: Kernel| k.unit().with_deps().map(|u| u.path).collect::<Vec<_>>();
        assert_eq!(
            paths(Kernel::MatmulPacked),
            ["linear/matmul_packed", "prim/half"]
        );
        assert_eq!(
            paths(Kernel::IndexTopk),
            ["attn/index_topk", "attn/index_cell"]
        );
        assert_eq!(paths(Kernel::Matmul), ["linear/matmul"]);
        // A comment naming another kernel brings nothing.
        assert_eq!(
            paths(Kernel::AttendPacked),
            ["attn/attend_packed", "prim/half"]
        );
    }
}
