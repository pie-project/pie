//! Hand-written kernels as modules (GENERIC.md, "Kernels as modules"): what sits outside a
//! kernels-ir `Region` (a mixer, a routed bank, a tiled conv), each one module under the poem-ir
//! group it implements, as kernels-metal and kernels-cuda lay theirs out, registered once in
//! [`KERNELS`]. A module states its CSL sources, the poem-ir ops it implements and, by regime,
//! how it runs; the compiler asks the registry instead of matching kernels by name. Elementwise
//! ops are not modules: they go through `fuse::chain` into a kernels-ir `Region`
//! (`emit::region`), the path every backend shares.
//!
//! Adding a kernel: a module file here (its sources, ops and the facets it has) and one line in
//! [`KERNELS`].

pub mod attn;
pub mod ssm;

/// A CSL source a module owns: its file name under `programs/wafer/ops` and its text.
pub type Source = (&'static str, &'static str);

/// A hand-written kernel.
pub trait Kernel: Sync {
    /// Its name: `<group>.<kernel>` (`attn.column`).
    fn name(&self) -> &'static str;
    /// The poem-ir ops it implements, by name.
    fn ops(&self) -> &'static [&'static str];
    /// The CSL op modules it owns (`programs/wafer/ops`), each owned by one kernel.
    fn sources(&self) -> &'static [Source];
}

/// Every kernel, in registry order: the order their sources are listed and digested in.
pub static KERNELS: &[&dyn Kernel] = &[&attn::column::COLUMN, &ssm::state::STATE];

/// The kernel that implements the poem-ir op `op`, if one does.
#[must_use]
pub fn by_op(op: &str) -> Option<&'static dyn Kernel> {
    KERNELS.iter().copied().find(|k| k.ops().contains(&op))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names, ops and sources are each a kernel's alone.
    #[test]
    fn a_name_an_op_or_a_source_is_one_kernels() {
        let mut names = std::collections::BTreeSet::new();
        let mut ops = std::collections::BTreeSet::new();
        let mut files = std::collections::BTreeSet::new();
        for k in KERNELS {
            assert!(names.insert(k.name()), "{}", k.name());
            for o in k.ops() {
                assert!(ops.insert(*o), "{o}: two kernels");
            }
            for (f, _) in k.sources() {
                assert!(files.insert(*f), "{f}: two kernels");
            }
        }
        assert_eq!(
            by_op("ssm.gated_delta").map(|k| k.name()),
            Some("ssm.state")
        );
    }
}
