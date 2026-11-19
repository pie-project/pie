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

use crate::emit::stream::HostArray;
use crate::floor::Coord;
use crate::floorplan::stream::KvColumn;
use crate::stream::build::Program;
use crate::stream::place::KvSched;
use crate::stream::unit::{Cols, Joint};

type R<T> = Result<T, String>;

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
    /// Its mixer on a unit's K / V column, where it runs one (the unit regime).
    fn column(&self) -> Option<&dyn Column> {
        None
    }
}

/// What sizes a K / V column's PEs ([`Column::pya`]).
#[derive(Clone, Copy, Debug)]
pub struct ColumnAsk {
    /// Q heads a stored head (the column's attention's, else the reading's).
    pub q_per_kv: u32,
    pub head_dim: u32,
    /// The positions it holds: a decode step's cache, else the step's stored rows.
    pub positions: u32,
    /// A decode step's rows (its lanes), 1 in a stream reading.
    pub lanes: u32,
    pub split: crate::stream::build::KvSplit,
}

/// A mixer on a unit's K / V column (P7, P8): its root takes each row's head (`[q | k | v]` and
/// [`Column::head_parts`]) off its unit, its PEs below hold what the rows leave (keys and values,
/// a state), each row's output goes back. The column's placement, chain and colours are the
/// floorplan's; what runs on it is the module's.
pub trait Column: Sync {
    /// Whether it runs the column of the attention (or recurrence) `j`.
    fn claims(&self, j: &Joint) -> bool;
    /// Its kind of region in a decode program's state root.
    fn region(&self, j: &Joint) -> &'static str;
    /// Whether its rows are a cache in pages (a decode step's), not a lane slot's state.
    fn paged(&self) -> bool;
    /// Whether its stored heads may be replicated over columns, a replica its share of the q
    /// heads (the search's choice where a head root's tiles do not fit).
    fn replicates(&self) -> bool;
    /// The PEs a column takes.
    fn pya(&self, ask: &ColumnAsk) -> u32;
    /// What a store sends past a head `h`'s `[q | k | v]`: (value, first column, width).
    fn head_parts(&self, _j: &Joint, _h: u32) -> Vec<(Cols, u32, u32)> {
        Vec::new()
    }
    /// Column `k`'s root file (name, text).
    fn root_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)>;
    /// Column `k`'s PEs' file (name, text).
    fn pe_file(&self, p: &Program, k: usize, ks: &KvSched) -> R<(String, String)>;
    /// The arrays the host writes into the column's root at `root`.
    fn root_arrays(&self, p: &Program, kc: &KvColumn, ks: &KvSched, root: Coord) -> Vec<HostArray>;
    /// The arrays the host writes into a PE of the column at `c` (past its config words).
    fn pe_arrays(&self, p: &Program, c: Coord) -> Vec<HostArray>;
}

/// The column that runs `j` (the reading's attention where a column has none of its own).
pub fn column(j: Option<&Joint>) -> R<&'static dyn Column> {
    let j = j.copied().unwrap_or(attn::column::UNSTATED);
    KERNELS
        .iter()
        .filter_map(|k| k.column())
        .find(|c| c.claims(&j))
        .ok_or_else(|| "a K / V column no kernel runs".to_string())
}

/// Every kernel, in registry order: the order their sources are listed and digested in.
pub static KERNELS: &[&dyn Kernel] = &[&attn::column::ATTENTION, &ssm::state::STATE];

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
